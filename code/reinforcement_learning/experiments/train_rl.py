from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

try:
    import gymnasium as gym
    _GymEnvBase = gym.Env
except Exception:
    gym = None
    _GymEnvBase = object

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except Exception:
    BaseFeaturesExtractor = nn.Module

CODE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from oui.core import compute_new_oui_with_stats
from oui.metrics import has_nonfinite, timing_summary_aliases

RESULTS_ROOT = REPO_ROOT / "descargas" / "reinforcement_learning" / "results"
LOGS_ROOT = REPO_ROOT / "descargas" / "reinforcement_learning" / "logs"
MILESTONE_PCTS = (1, 2, 5, 10, 15, 20, 30, 50, 75, 100)
RUN_STATE: dict[str, Any] = {"step": None, "last_eval": None}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def read_yaml(path):
    with Path(path).open("r", encoding="utf-8") as h:
        return yaml.safe_load(h) or {}


def deep_update(base, updates):
    out = dict(base)
    for k, v in updates.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def make_jsonable(v):
    if isinstance(v, dict):
        return {str(k): make_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [make_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().tolist()
    return v


def write_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_jsonable(payload), indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as h:
        h.write(json.dumps(make_jsonable(payload), sort_keys=False) + "\n")


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def output_dir_for(cfg, args):
    if args.output_dir:
        return Path(args.output_dir).resolve()
    if args.smoke:
        return RESULTS_ROOT / "smoke" / cfg["experiment_id"] / timestamp()
    return RESULTS_ROOT / "full" / cfg["experiment_id"] / f"lr_{args.lr:.1e}__seed_{args.seed}"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MinAtarGymWrapper(_GymEnvBase):
    def __init__(self, game: str, max_episode_steps: int = 5000):
        import gymnasium as gym_local
        try:
            from minatar import Environment
        except Exception:
            from minatar.environment import Environment

        super().__init__()
        self.env = Environment(game)
        state = self.env.state()
        self.observation_space = gym_local.spaces.Box(0.0, 1.0, shape=(state.shape[2], state.shape[0], state.shape[1]), dtype=np.float32)
        self.action_space = gym_local.spaces.Discrete(self.env.num_actions())
        self.metadata = {}
        self.render_mode = None
        self.spec = None
        self.max_episode_steps = int(max_episode_steps)
        self.elapsed_steps = 0

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.env.reset()
        self.elapsed_steps = 0
        return self._obs(), {}

    def step(self, action):
        reward, terminal = self.env.act(int(action))
        self.elapsed_steps += 1
        truncated = self.elapsed_steps >= self.max_episode_steps
        return self._obs(), float(reward), bool(terminal), bool(truncated), {}

    def _obs(self):
        return np.transpose(self.env.state().astype(np.float32), (2, 0, 1))

    def render(self):
        return None

    def close(self):
        return None


def make_env_fn(cfg, seed: int, rank: int = 0):
    def thunk():
        import gymnasium as gym

        env_id = cfg["env"]["id"]
        if cfg["env"].get("kind") == "minigrid":
            import minigrid  # noqa: F401
            from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgPartialObsWrapper
            from gymnasium.wrappers import TransformObservation

            env = gym.make(env_id)
            if bool(cfg["env"].get("fully_observed", False)):
                env = FullyObsWrapper(env)
            else:
                env = RGBImgPartialObsWrapper(env)
            env = ImgObsWrapper(env)
            obs_shape = env.observation_space.shape
            env = TransformObservation(
                env,
                lambda obs: np.transpose(obs, (2, 0, 1)),
                observation_space=gym.spaces.Box(0, 255, shape=(obs_shape[2], obs_shape[0], obs_shape[1]), dtype=np.uint8),
            )
        elif cfg["env"].get("kind") == "minatar":
            env = MinAtarGymWrapper(cfg["env"]["game"], int(cfg["env"].get("max_episode_steps", 5000)))
        else:
            env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env
    return thunk


def minigrid_wrapper_metadata(cfg) -> dict[str, Any]:
    if cfg["env"].get("kind") != "minigrid":
        return {}
    wrappers = ["FullyObsWrapper" if bool(cfg["env"].get("fully_observed", False)) else "RGBImgPartialObsWrapper", "ImgObsWrapper", "channel_first_TransformObservation"]
    return {"wrappers": wrappers, "fully_observed": bool(cfg["env"].get("fully_observed", False))}


class MiniGridExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        BaseFeaturesExtractor.__init__(self, observation_space, features_dim)
        c = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Conv2d(c, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.net(torch.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())

    def forward(self, x):
        return self.linear(self.net(x.float() / 255.0))


class MinAtarExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        BaseFeaturesExtractor.__init__(self, observation_space, features_dim)
        c = observation_space.shape[0]
        self.net = nn.Sequential(nn.Conv2d(c, 16, 3, padding=1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n = self.net(torch.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())

    def forward(self, x):
        return self.linear(self.net(x.float()))


class HookedOUI:
    def __init__(self, model: nn.Module, patterns: list[str]):
        self.records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.enabled = False
        self.handles = []
        depth = 0
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            if patterns and not any(p in name for p in patterns):
                continue
            key = name.replace(".", "_") or f"module_{depth}"
            self.records[key] = {"name": key, "depth": depth, "module_type": type(module).__name__, "source_name": name}
            self.handles.append(module.register_forward_hook(self._hook(key, depth, type(module).__name__)))
            depth += 1

    def _hook(self, key, depth, module_type):
        def fn(_m, _i, out):
            if not self.enabled or not torch.is_tensor(out):
                return
            start = time.perf_counter()
            oui, stats = compute_new_oui_with_stats(out.detach())
            self.records[key] = {
                **self.records[key],
                "oui": oui,
                "activation_shape": tuple(out.shape),
                "feature_dim": stats.feature_dim,
                "batch_size": stats.batch_size,
                "working_set_bytes": stats.estimated_working_set_bytes,
                "compute_seconds": time.perf_counter() - start,
            }
        return fn

    def capture(self):
        class Ctx:
            def __init__(self, owner):
                self.owner = owner
            def __enter__(self):
                self.owner.enabled = True
                for r in self.owner.records.values():
                    for k in ("oui", "activation_shape", "feature_dim", "batch_size", "working_set_bytes", "compute_seconds"):
                        r.pop(k, None)
                return self.owner
            def __exit__(self, *_):
                self.owner.enabled = False
                return False
        return Ctx(self)

    def rows_agg(self):
        rows = [dict(r) for r in self.records.values() if "oui" in r]
        vals = np.asarray([r["oui"] for r in rows], dtype=np.float64)
        agg = {"oui_mean": math.nan, "oui_std": math.nan, "oui_min": math.nan, "oui_max": math.nan}
        if len(vals):
            agg = {"oui_mean": float(vals.mean()), "oui_std": float(vals.std()), "oui_min": float(vals.min()), "oui_max": float(vals.max())}
        for group, markers in {
            "backbone": ("features_extractor",),
            "actor": ("policy_net", "action_net", "pi_features_extractor", "actor", "latent_pi"),
            "critic": ("value_net", "vf_features_extractor", "critic", "qf", "latent_vf"),
            "q_network": ("q_net", "q_net_target"),
        }.items():
            group_vals = np.asarray(
                [r["oui"] for r in rows if any(marker in str(r.get("source_name", "")) for marker in markers)],
                dtype=np.float64,
            )
            if len(group_vals):
                agg[f"oui_{group}_mean"] = float(group_vals.mean())
                agg[f"oui_{group}_std"] = float(group_vals.std())
                agg[f"oui_{group}_min"] = float(group_vals.min())
                agg[f"oui_{group}_max"] = float(group_vals.max())
        agg["oui_compute_seconds"] = sum(float(r.get("compute_seconds", 0.0)) for r in rows)
        return rows, agg

    def close(self):
        for h in self.handles:
            h.remove()


def param_norm(model):
    total = torch.zeros((), device=next(model.parameters()).device)
    for p in model.parameters():
        if p.requires_grad:
            total += p.detach().float().pow(2).sum()
    return float(total.sqrt().item())


def current_grad_norm(model):
    total = None
    for p in model.parameters():
        if p.grad is None:
            continue
        val = p.grad.detach().float().pow(2).sum()
        total = val if total is None else total + val
    return math.nan if total is None else float(total.sqrt().item())


def policy_probe_stats(model, states: np.ndarray, device) -> dict[str, float]:
    obs = torch.as_tensor(states[: min(256, len(states))], device=device)
    out: dict[str, float] = {}
    with torch.no_grad():
        try:
            dist = model.policy.get_distribution(obs)
            distribution = getattr(dist, "distribution", None)
            if hasattr(distribution, "logits"):
                logits = distribution.logits.detach().float()
                out.update(
                    {
                        "actor_logits_mean": float(logits.mean().item()),
                        "actor_logits_std": float(logits.std(unbiased=False).item()),
                        "actor_logits_min": float(logits.min().item()),
                        "actor_logits_max": float(logits.max().item()),
                    }
                )
        except Exception:
            pass
        try:
            values = model.policy.predict_values(obs).detach().float().flatten()
            out.update(
                {
                    "value_prediction_mean": float(values.mean().item()),
                    "value_prediction_std": float(values.std(unbiased=False).item()),
                    "value_prediction_min": float(values.min().item()),
                    "value_prediction_max": float(values.max().item()),
                }
            )
        except Exception:
            pass
    return out


def metric_steps(total, interval):
    steps = {interval, total}
    steps.update(range(interval, total + 1, interval))
    for pct in MILESTONE_PCTS:
        steps.add(max(1, round(total * pct / 100)))
    return steps


def collect_probe(env_fn, n_states, out_dir: Path, seed: int):
    env = env_fn()
    obs, _ = env.reset(seed=seed)
    states = []
    while len(states) < n_states:
        states.append(obs)
        obs, _r, term, trunc, _info = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset()
    arr = np.asarray(states)
    np.savez_compressed(out_dir / "probe_states.npz", states=arr)
    write_json(
        out_dir / "probe_metadata.json",
        {
            "num_states": int(len(arr)),
            "shape": list(arr.shape),
            "seed": seed,
            "observation_mean": float(arr.mean()),
            "observation_std": float(arr.std()),
            "observation_min": float(arr.min()),
            "observation_max": float(arr.max()),
        },
    )
    env.close()
    return arr


def evaluate_random_policy(env_fn, out_dir: Path, seed: int, n_episodes: int = 20):
    env = env_fn()
    returns, lengths = [], []
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total = 0.0
        length = 0
        while not done:
            obs, reward, terminated, truncated, _info = env.step(env.action_space.sample())
            total += float(reward)
            length += 1
            done = bool(terminated or truncated)
        returns.append(total)
        lengths.append(length)
    env.close()
    row = {
        "seed": seed,
        "n_episodes": n_episodes,
        "random_policy_returns": returns,
        "random_policy_lengths": lengths,
        "random_policy_mean_return": float(np.mean(returns)),
        "random_policy_std_return": float(np.std(returns)),
        "random_policy_min_return": float(np.min(returns)),
        "random_policy_max_return": float(np.max(returns)),
    }
    write_json(out_dir / "random_policy_eval.json", row)
    return row


def policy_predict_oui(model, oui: HookedOUI, states: np.ndarray, device, out_dir: Path, step: int, phase: str):
    total_start = time.perf_counter()
    obs = torch.as_tensor(states[: min(256, len(states))], device=device)
    with torch.no_grad(), oui.capture():
        _ = model.policy(obs)
    rows, agg = oui.rows_agg()
    agg[f"oui_{phase}_total_seconds"] = time.perf_counter() - total_start
    for row in rows:
        append_jsonl(out_dir / f"oui_{phase}.jsonl", {"record_type": "oui_module", "phase": phase, "timestep": step, **row})
    append_jsonl(out_dir / f"oui_{phase}.jsonl", {"record_type": "aggregate", "phase": phase, "timestep": step, **agg})
    return agg


def build_model(cfg, env, lr):
    from stable_baselines3 import DQN, PPO, SAC

    algo = cfg["algorithm"]
    common = {"learning_rate": lr, "seed": cfg["seed"], "verbose": 0, "tensorboard_log": None}
    if algo == "PPO":
        policy_kwargs = dict(activation_fn=getattr(nn, cfg["policy"].get("activation", "Tanh")), net_arch=cfg["policy"].get("net_arch"))
        if cfg["env"].get("kind") == "minigrid":
            policy_kwargs = dict(features_extractor_class=MiniGridExtractor, features_extractor_kwargs=dict(features_dim=128), net_arch=dict(pi=[], vf=[]))
            policy = "CnnPolicy"
        else:
            policy = "MlpPolicy"
        return PPO(policy, env, policy_kwargs=policy_kwargs, **cfg["hyperparams"], **common)
    if algo == "DQN":
        policy_kwargs = dict(features_extractor_class=MinAtarExtractor, features_extractor_kwargs=dict(features_dim=128), net_arch=[])
        return DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, **cfg["hyperparams"], **common)
    if algo == "SAC":
        policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=cfg["policy"].get("net_arch", [256, 256]))
        return SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, **cfg["hyperparams"], **common)
    raise ValueError(f"Unsupported algorithm: {algo}")


class OUICallback:
    def __init__(self, cfg, out_dir: Path, probe_states, start_wall, timing_totals: dict[str, float], timing_counts: dict[str, int]):
        from stable_baselines3.common.callbacks import BaseCallback

        class Callback(BaseCallback):
            def __init__(self):
                super().__init__(verbose=0)
                self.oui = None
                self.last_params = None

            def _on_training_start(self):
                patterns = cfg.get("oui_patterns", [])
                self.oui = HookedOUI(self.model.policy, patterns)
                self.last_params = [p.detach().clone() for p in self.model.policy.parameters() if p.requires_grad]

            def _on_step(self):
                step = int(self.num_timesteps)
                RUN_STATE["step"] = step
                interval = int(cfg["tracking"]["metric_interval"])
                if step % interval != 0 and step not in cfg["tracking"]["metric_steps"]:
                    return True
                device = self.model.device
                before = self.last_params or []
                update = math.nan
                if before:
                    total = torch.zeros((), device=device)
                    idx = 0
                    for p in self.model.policy.parameters():
                        if p.requires_grad:
                            total += (p.detach() - before[idx]).float().pow(2).sum()
                            idx += 1
                    update = float(total.sqrt().item())
                self.last_params = [p.detach().clone() for p in self.model.policy.parameters() if p.requires_grad]
                pnorm = param_norm(self.model.policy)
                train_agg: dict[str, Any] = {}
                train_obs = self.locals.get("new_obs", self.locals.get("obs"))
                if train_obs is not None:
                    train_agg = policy_predict_oui(self.model, self.oui, np.asarray(train_obs), device, out_dir, step, "train")
                    timing_totals["oui_train_seconds"] += float(
                        train_agg.get("oui_train_total_seconds", train_agg.get("oui_compute_seconds", 0.0)) or 0.0
                    )
                    timing_counts["oui_train"] += 1
                probe_agg = policy_predict_oui(self.model, self.oui, probe_states, device, out_dir, step, "probe")
                timing_totals["oui_probe_seconds"] += float(
                    probe_agg.get("oui_probe_total_seconds", probe_agg.get("oui_compute_seconds", 0.0)) or 0.0
                )
                timing_counts["oui_probe"] += 1
                metrics = {
                    "record_type": "train_metrics",
                    "timestep": step,
                    "wall_time": time.time() - start_wall,
                    "current_lr": self.model.lr_schedule(self.model._current_progress_remaining),
                    "configured_lr": cfg["lr"],
                    "configured_wd": None,
                    "grad_norm_pre_clip": current_grad_norm(self.model.policy),
                    "grad_norm_post_clip": math.nan,
                    "clipping_applied": None,
                    "param_norm": pnorm,
                    "update_norm": update,
                    "update_to_weight_ratio": update / max(pnorm, 1e-12) if math.isfinite(update) else math.nan,
                    **{f"train_{k}": v for k, v in train_agg.items()},
                    **{f"probe_{k}": v for k, v in probe_agg.items()},
                    **policy_probe_stats(self.model, probe_states, device),
                }
                for key, val in self.model.logger.name_to_value.items():
                    if isinstance(val, (int, float, np.number)):
                        metrics[key.replace("/", "_")] = float(val)
                append_jsonl(out_dir / "metrics.jsonl", metrics)
                append_jsonl(
                    out_dir / "overhead.jsonl",
                    {
                        "record_type": "metric_overhead",
                        "timestep": step,
                        "wall_time": time.time() - start_wall,
                        "oui_train_total_seconds": train_agg.get("oui_train_total_seconds"),
                        "oui_train_compute_seconds": train_agg.get("oui_compute_seconds"),
                        "oui_probe_total_seconds": probe_agg.get("oui_probe_total_seconds"),
                        "oui_probe_compute_seconds": probe_agg.get("oui_compute_seconds"),
                    },
                )
                return True

            def _on_training_end(self):
                if self.oui is not None:
                    self.oui.close()

        self.cls = Callback


def evaluate_policy_to_json(model, env_fn, out_dir: Path, step: int, n_episodes: int, eval_seed: int):
    from stable_baselines3.common.evaluation import evaluate_policy

    env = env_fn()
    eval_start = time.perf_counter()
    returns, lengths = evaluate_policy(model, env, n_eval_episodes=n_episodes, deterministic=True, return_episode_rewards=True, warn=False)
    eval_seconds = time.perf_counter() - eval_start
    env.close()
    row = {
        "record_type": "eval",
        "timestep": step,
        "eval_mean_return": float(np.mean(returns)),
        "eval_std_return": float(np.std(returns)),
        "eval_min_return": float(np.min(returns)),
        "eval_max_return": float(np.max(returns)),
        "eval_episode_returns": [float(x) for x in returns],
        "eval_episode_lengths": [int(x) for x in lengths],
        "eval_seed": int(eval_seed),
        "eval_seconds": eval_seconds,
    }
    append_jsonl(out_dir / "eval.jsonl", row)
    RUN_STATE["last_eval"] = row
    return row


def main_impl():
    args = parse_args()
    cfg = read_yaml(args.config)
    if args.smoke:
        cfg = deep_update(cfg, cfg.get("smoke", {}))
    cfg["seed"] = args.seed
    cfg["lr"] = args.lr
    set_seed(args.seed)
    out_dir = output_dir_for(cfg, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", {"config": cfg, "lr": args.lr, "seed": args.seed, "smoke": args.smoke})

    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except Exception as exc:
        raise ImportError('RL requires: pip install stable-baselines3 gymnasium minigrid minatar; R1 also needs box2d-py, R5/R6 need mujoco.') from exc

    total_timesteps = int(cfg["training"]["total_timesteps"])
    metric_interval = max(1000, total_timesteps // 1000)
    eval_interval = max(1, total_timesteps // 20)
    cfg["tracking"]["metric_interval"] = metric_interval
    cfg["tracking"]["metric_steps"] = sorted(metric_steps(total_timesteps, metric_interval))
    env_fns = [make_env_fn(cfg, args.seed, i) for i in range(int(cfg["env"].get("n_envs", 1)))]
    VecCls = SubprocVecEnv if len(env_fns) > 1 and not args.smoke else DummyVecEnv
    env = VecCls(env_fns)
    eval_env_fn = make_env_fn(cfg, args.seed + 10000, 0)
    eval_seed = args.seed + 10000
    single_env = eval_env_fn()
    write_json(
        out_dir / "env_metadata.json",
        {
            "env_id": cfg["env"]["id"],
            "kind": cfg["env"].get("kind"),
            "observation_space": str(single_env.observation_space),
            "action_space": str(single_env.action_space),
            **minigrid_wrapper_metadata(cfg),
        },
    )
    single_env.close()
    random_policy = evaluate_random_policy(eval_env_fn, out_dir, args.seed + 30000, int(cfg["tracking"].get("random_policy_episodes", 20)))
    probe_states = collect_probe(eval_env_fn, int(cfg["tracking"].get("probe_states", 2048)), out_dir, args.seed + 20000)
    model = build_model(cfg, env, args.lr)
    start_wall = time.time()
    timing_totals = {
        "learn_seconds": 0.0,
        "oui_train_seconds": 0.0,
        "oui_probe_seconds": 0.0,
        "eval_seconds": 0.0,
        "other_overhead_seconds": 0.0,
    }
    timing_counts = {"learn_chunks": 0, "oui_train": 0, "oui_probe": 0, "eval": 0}
    cb = OUICallback(cfg, out_dir, probe_states, start_wall, timing_totals, timing_counts).cls()
    eval_steps = sorted(metric_steps(total_timesteps, eval_interval))

    best_eval = -float("inf")
    best_step = -1
    for target in eval_steps:
        delta = target - int(model.num_timesteps)
        if delta > 0:
            learn_start = time.perf_counter()
            model.learn(total_timesteps=delta, reset_num_timesteps=False, callback=cb, progress_bar=False)
            timing_totals["learn_seconds"] += time.perf_counter() - learn_start
            timing_counts["learn_chunks"] += 1
        row = evaluate_policy_to_json(model, eval_env_fn, out_dir, int(model.num_timesteps), int(cfg["tracking"].get("eval_episodes", 10)), eval_seed)
        timing_totals["eval_seconds"] += float(row.get("eval_seconds", 0.0) or 0.0)
        timing_counts["eval"] += 1
        if row["eval_mean_return"] > best_eval:
            best_eval = row["eval_mean_return"]
            best_step = int(model.num_timesteps)
    if int(model.num_timesteps) < total_timesteps:
        learn_start = time.perf_counter()
        model.learn(total_timesteps=total_timesteps - int(model.num_timesteps), reset_num_timesteps=False, callback=cb, progress_bar=False)
        timing_totals["learn_seconds"] += time.perf_counter() - learn_start
        timing_counts["learn_chunks"] += 1
    final_eval = evaluate_policy_to_json(model, eval_env_fn, out_dir, int(model.num_timesteps), int(cfg["tracking"].get("eval_episodes", 10)), eval_seed)
    timing_totals["eval_seconds"] += float(final_eval.get("eval_seconds", 0.0) or 0.0)
    timing_counts["eval"] += 1
    best_eval = max(best_eval, final_eval["eval_mean_return"])
    if final_eval["eval_mean_return"] >= best_eval:
        best_step = int(model.num_timesteps)
    total_time = time.time() - start_wall
    measured = sum(v for k, v in timing_totals.items() if k != "other_overhead_seconds")
    timing_totals["other_overhead_seconds"] = max(0.0, total_time - measured)
    timing_percentages = {k.replace("_seconds", "_pct"): (v / total_time * 100.0 if total_time > 0 else 0.0) for k, v in timing_totals.items()}
    timing_averages = {
        "learn_avg_seconds": timing_totals["learn_seconds"] / max(1, timing_counts["learn_chunks"]),
        "oui_train_avg_seconds": timing_totals["oui_train_seconds"] / max(1, timing_counts["oui_train"]),
        "oui_probe_avg_seconds": timing_totals["oui_probe_seconds"] / max(1, timing_counts["oui_probe"]),
        "eval_avg_seconds": timing_totals["eval_seconds"] / max(1, timing_counts["eval"]),
    }
    summary = {
            "failed_run": False,
            "experiment_id": cfg["experiment_id"],
            "algorithm": cfg["algorithm"],
            "env_id": cfg["env"]["id"],
            "lr": args.lr,
            "seed": args.seed,
            "final_step": int(model.num_timesteps),
            "total_timesteps": total_timesteps,
            "last_eval_return": final_eval["eval_mean_return"],
            "best_eval_return": best_eval,
            "best_eval_mean_return": best_eval,
            "best_eval_step": best_step,
            "final_eval_return": final_eval["eval_mean_return"],
            "final_eval_mean_return": final_eval["eval_mean_return"],
            "final_eval_std_return": final_eval["eval_std_return"],
            "final_eval_min_return": final_eval["eval_min_return"],
            "final_eval_max_return": final_eval["eval_max_return"],
            "eval_seed": eval_seed,
            "random_policy_mean_return": random_policy.get("random_policy_mean_return"),
            "random_policy_std_return": random_policy.get("random_policy_std_return"),
            "random_policy_min_return": random_policy.get("random_policy_min_return"),
            "random_policy_max_return": random_policy.get("random_policy_max_return"),
            "wall_time": total_time,
            **timing_summary_aliases(timing_totals, total_time),
            "timing_seconds": timing_totals,
            "timing_percentages": timing_percentages,
            "timing_counts": timing_counts,
            "timing_averages": timing_averages,
            "integrity": {
                "run_completed": int(model.num_timesteps) >= total_timesteps,
                "nonfinite_in_summary": False,
                "num_train_checkpoints": timing_counts["oui_probe"],
                "num_metric_checkpoints": timing_counts["oui_probe"],
                "num_evaluations": timing_counts["eval"],
                "num_oui_train_measurements": timing_counts["oui_train"],
                "num_oui_probe_measurements": timing_counts["oui_probe"],
            },
        }
    summary["integrity"]["nonfinite_in_summary"] = has_nonfinite(summary)
    write_json(out_dir / "summary.json", summary)
    env.close()


def main():
    try:
        main_impl()
    except Exception as exc:
        args = parse_args()
        cfg = read_yaml(args.config)
        if args.smoke:
            cfg = deep_update(cfg, cfg.get("smoke", {}))
        out_dir = output_dir_for(cfg, args)
        write_json(
            out_dir / "failed_run.json",
            {
                "failed_run": True,
                "experiment_id": cfg.get("experiment_id"),
                "lr": args.lr,
                "seed": args.seed,
                "step": RUN_STATE.get("step"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=10),
                "last_eval": RUN_STATE.get("last_eval"),
            },
        )
        raise


if __name__ == "__main__":
    main()
