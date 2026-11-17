"""Small CPU study of utilization, transmitted responses and local bias adaptation.

This study tests a conditional mechanism. Its protocol is independent of selector
coefficient fitting and uses no archived benchmark trajectories.
"""
from pathlib import Path
import argparse,json,time,copy
import numpy as np
import pandas as pd
import torch
from torch import nn

class Net(nn.Module):
    def __init__(self,activation):
        super().__init__();self.first=nn.Linear(4,8);self.second=nn.Linear(8,8);self.out=nn.Linear(8,1)
        self.act=nn.ReLU() if activation=='relu' else nn.GELU()
    def forward(self,x):
        self.z1=self.first(x);self.h1=self.act(self.z1)
        self.z2=self.second(self.h1);self.h2=self.act(self.z2)
        return self.out(self.h2).squeeze(-1)

def measure(net,x,y):
    f=net(x);r=(y-f).detach().double();r=r-r.mean()
    z=[net.z1,net.z2];h=[net.h1,net.h2]
    derivatives=torch.autograd.grad(f.sum(),z+h,retain_graph=True)
    t=torch.cat(derivatives[:2],dim=1).detach().double()
    phi=t-t.mean(0);q=phi.T@phi/len(x);eig=torch.linalg.eigvalsh(q)
    c=phi.T@r/len(x);alpha=.05/max(float(eig[-1]),1e-12)
    p=[a.detach().gt(0).double().mean(0) for a in z]
    u=[float((2*torch.minimum(a,1-a)).mean()) for a in p]
    G=float(np.exp(np.log(np.array(u)+1e-8).mean()))
    pred=.5*float(r.square().mean()-(r-phi@(alpha*c)).square().mean())
    trial=Net('relu' if isinstance(net.act,nn.ReLU) else 'gelu');trial.load_state_dict(net.state_dict())
    with torch.no_grad():
        trial.first.bias.add_((alpha*c[:8]).to(trial.first.bias))
        trial.second.bias.add_((alpha*c[8:]).to(trial.second.bias))
        delta=(trial(x)-f.detach()).double();delta=delta-delta.mean()
        actual=.5*float(r.square().mean()-(r-delta).square().mean())
        flip=float(torch.cat([a.detach().gt(0) for a in z],1).ne(torch.cat([trial.z1.gt(0),trial.z2.gt(0)],1)).double().mean())
    var=t.var(0,unbiased=False)
    bounds=[];contrast=[];certificates=[]
    for layer,(probs,T,D) in enumerate(zip(p,derivatives[:2],derivatives[2:])):
        gates=z[layer].detach().gt(0);T=T.detach().double();D=D.detach().double()
        for j in range(8):
            mask=gates[:,j];pj=float(probs[j])
            if 0<pj<1:
                meanactive=float(T[mask,j].mean());meaninactive=float(T[~mask,j].mean())
                contrast.append(pj*(1-pj)*(meanactive-meaninactive)**2)
                if isinstance(net.act,nn.ReLU):
                    a2=float(D[mask,j].square().mean());bounds.append((pj*(1-pj)*a2,float(T[:,j].var(unbiased=False))))
        if isinstance(net.act,nn.ReLU) and u[layer]>0:
            unit_u=2*torch.minimum(probs,1-probs)
            selected=torch.where(unit_u>=u[layer]/2)[0]
            centered=T[:,selected]-T[:,selected].mean(0)
            gram=centered.T@centered/len(x);diag=torch.diag(gram)
            a2=min(float(D[gates[:,j],j].square().mean()) for j in selected)
            if bool((diag>1e-14).all()):
                correlation=gram/torch.sqrt(diag[:,None]*diag[None,:])
                kappa=max(0.,float(torch.linalg.eigvalsh(correlation)[0]))
                lower=kappa*a2*u[layer]/8
                observed=float(torch.linalg.eigvalsh(gram)[0])
                certificates.append((lower,observed,len(selected)/8,u[layer]/(2-u[layer])))
    # Preserve the hidden codes and vary only transmission to the output.
    transmissions=[]
    for scale in [0.,.5,1.]:
        other=Net('relu' if isinstance(net.act,nn.ReLU) else 'gelu');other.load_state_dict(net.state_dict())
        with torch.no_grad():other.out.weight.mul_(scale)
        otherf=other(x)
        dt=torch.autograd.grad(otherf.sum(),[other.z1,other.z2])
        other_t=torch.cat(dt,1).detach().double()
        transmissions.append(float(other_t.var(0,unbiased=False).sum()))
    energy=float(var.sum())
    transmission_error=max(abs(transmissions[i]-energy*s*s) for i,s in enumerate([0.,.5,1.]))
    return dict(G=G,mean_U=float(np.mean(u)),response_energy=energy,gram_rank=int((eig>max(float(eig[-1])*1e-8,1e-12)).sum()),
        gram_min=float(eig[0]),gram_max=float(eig[-1]),alignment=float(c.square().sum()),
        predicted_gain=pred,actual_gain=actual,gate_change_fraction=flip,
        response_contrast_bound=float(sum(contrast)),transmission_scaling_error=transmission_error,
        module_certificate_checks=len(certificates),module_positive_certificates=sum(a>1e-10 for a,b,c,d in certificates),
        module_certificate_violations=sum(b+1e-10<a or c+1e-10<d for a,b,c,d in certificates),
        relu_bound_violations=sum(v+1e-10<b for b,v in bounds),relu_bound_coordinates=len(bounds))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/mechanism_rerun')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    generator=torch.Generator().manual_seed(270927)
    train=torch.randn(512,4,generator=generator);probe=torch.randn(256,4,generator=generator)
    target=lambda x:torch.sin(x[:,0])+.5*x[:,1].square()+.25*x[:,2]
    train_y=target(train);probe_y=target(probe)
    rows=[];start=time.perf_counter()
    for activation in ['relu','gelu']:
        for lr in [.001,.01,.03]:
            for wd in [0.,.01]:
                for seed in [0,1,2]:
                    torch.manual_seed(seed);net=Net(activation);opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=wd)
                    for step in range(121):
                        if step in [6,12,24,120]:
                            row=measure(net,probe,probe_y);row.update(activation=activation,lr=lr,wd=wd,seed=seed,step=step);rows.append(row)
                        if step==120:break
                        idx=torch.randperm(512)[:64];loss=.5*(net(train[idx])-train_y[idx]).square().mean()
                        opt.zero_grad();loss.backward();opt.step()
    d=pd.DataFrame(rows);d.to_csv(args.output/'measurements.csv',index=False)
    summary=[]
    for activation,g in d.groupby('activation'):
        positive=g[g.predicted_gain>1e-10]
        relative=(positive.actual_gain-positive.predicted_gain).abs()/positive.predicted_gain
        summary.append(dict(activation=activation,checkpoints=len(g),runs=18,
            mean_G=float(g.G.mean()),mean_response_energy=float(g.response_energy.mean()),
            median_rank=float(g.gram_rank.median()),predicted_actual_correlation=float(g.predicted_gain.corr(g.actual_gain)),
            median_relative_gain_error=float(relative.median()),p90_relative_gain_error=float(relative.quantile(.9)),
            median_gate_change=float(g.gate_change_fraction.median()),
            positive_actual_gain=int(g.actual_gain.gt(0).sum()),relu_bound_coordinates=int(g.relu_bound_coordinates.sum()),
            relu_bound_violations=int(g.relu_bound_violations.sum()),max_transmission_error=float(g.transmission_scaling_error.max())))
    pd.DataFrame(summary).to_csv(args.output/'summary.csv',index=False)
    protocol=dict(purpose='Conditional mechanism illustration; separate from selector validation and weight fitting',
        activations=['ReLU','GELU'],architecture=[4,8,8,1],training_inputs=512,probe_inputs=256,
        target='sin(x0)+0.5*x1^2+0.25*x2; iid N(0,I4) inputs',data_seed=270927,
        lr=[.001,.01,.03],wd=[0.,.01],seeds=[0,1,2],optimizer='AdamW defaults',steps=120,batch=64,
        checkpoints=[6,12,24,120],local_step='0.05 / lambda_max(Q)',bias_coordinates=16,
        output_weight_scales=[0.,.5,1.],elapsed_seconds=time.perf_counter()-start,
        torch_version=torch.__version__,numpy_version=np.__version__,device='CPU',threads=2)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    assert d.relu_bound_violations.sum()==0
    assert d.module_certificate_violations.sum()==0
    assert d.transmission_scaling_error.max()<1e-7
    print(pd.DataFrame(summary).to_string(index=False));print('CPU seconds',protocol['elapsed_seconds'])

if __name__=='__main__':main()
