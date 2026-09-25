"""Colab: python smoke_test.py config/colab.yaml. Tiny model, one backward pass."""
import sys, ast
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from train import make_model, make_dataset, batch_loss

with open(sys.argv[1]) as f: conf=yaml.safe_load(f)
dataset=make_dataset(conf,'val')
print('Dictionary vocabulary:',len(dataset.event2idx), 'including PAD:',dataset.vocab_size)
for name in ('enc_n_layer','dec_n_layer'): conf['model'][name]=1
for name in ('enc_n_head','dec_n_head'): conf['model'][name]=4
for name in ('enc_d_model','dec_d_model','d_embed'): conf['model'][name]=32
for name in ('enc_d_ff','dec_d_ff'): conf['model'][name]=64
conf['model']['d_latent']=16
for name in ('d_tempo_emb','d_density_emb','d_velocity_emb'): conf['model'][name]=8
model=make_model(conf,dataset.vocab_size)
batch=next(iter(DataLoader(dataset,batch_size=1)))
for name in ('tempo','density','velocity'):
    n=int(batch['enc_n_bars'][0]);start,end=batch['bar_pos'][0,n-1:n+1]
    assert (batch[name+'_cls'][0,start:end]==batch[name+'_cls_bar'][0,n-1]).all()
loss=batch_loss(model,batch,'cpu',.1,.25)
assert torch.isfinite(loss['total_loss'])
loss['total_loss'].backward()
for name in ('tempo','density','velocity'):
    grad=getattr(model,name+'_attr_emb').emb_lookup.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
print('PASS: final-bar conditions, forward/loss/backward, three embedding gradients')
# Execute pure generation helpers without the command-line entry point.
source=Path(__file__).with_name('generate.py').read_text(); tree=ast.parse(source)
names={'get_beat_idx','temperatured_softmax','nucleus','generate_on_latent_ctrl_vanilla_truncate'}
funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
import time
from copy import deepcopy
from scipy.stats import entropy
from utils import numpy_to_tensor, tensor_to_numpy
ns=dict(np=np,torch=torch,time=time,deepcopy=deepcopy,entropy=entropy,device='cpu',
        numpy_to_tensor=numpy_to_tensor,tensor_to_numpy=tensor_to_numpy)
exec(compile(ast.Module(body=funcs,type_ignores=[]),'generate_helpers','exec'),ns)
vocab={'Bar_None':0,'Beat_0':1,'Note_Pitch_60':2,'Note_Velocity_60':3,'Note_Duration_120':4,'EOS_None':5}
class FakeModel:
    def __init__(self):self.calls=[];self.step=0
    def generate(self,inp,latent,tempo,density,velocity):
        self.calls.append((inp.clone(),tempo.clone(),density.clone(),velocity.clone()))
        sequence=[1,2,3,4,0,1,2,3,4,5]
        logits=torch.full((1,7),-float('inf'));logits[0,sequence[self.step]]=0.;self.step+=1;return logits
fake=FakeModel()
song,_,_=ns['generate_on_latent_ctrl_vanilla_truncate'](fake,torch.zeros(2,16),torch.tensor([1,2]),
    torch.tensor([3,4]),torch.tensor([5,8]),vocab,{v:k for k,v in vocab.items()},
    max_events=64,max_input_len=4,truncate_len=2)
assert song==[0,1,2,3,4,0,1,2,3,4]
assert [int(c[1][-1,0]) for c in fake.calls]==[1]*5+[2]*5
assert int(fake.calls[-1][3][-1,0])==8
print('PASS: bar switching, context truncation, velocity missing class, EOS preserves final note')
