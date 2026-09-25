"""Colab: python train.py config/colab.yaml [--resume]. One epoch initially."""
import os, sys, time, argparse, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model'))
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import yaml
from dataloader import REMIFullSongTransformerDataset, pickle_load
from model.musemorphose import MuseMorphose


def make_model(conf, n_token):
    m = conf['model']
    return MuseMorphose(
        m['enc_n_layer'], m['enc_n_head'], m['enc_d_model'], m['enc_d_ff'],
        m['dec_n_layer'], m['dec_n_head'], m['dec_d_model'], m['dec_d_ff'],
        m['d_latent'], m['d_embed'], n_token,
        **{key:m[key] for key in ('d_tempo_emb','d_density_emb','d_velocity_emb',
                                  'n_tempo_cls','n_density_cls','n_velocity_cls','cond_mode')})


def make_dataset(conf, split):
    d = conf['data']
    return REMIFullSongTransformerDataset(
        d['data_dir'], d['vocab_path'], model_enc_seqlen=d['enc_seqlen'],
        model_dec_seqlen=d['dec_seqlen'], model_max_bars=d['max_bars'],
        pieces=pickle_load(d[split+'_split']), do_augment=False,
        appoint_st_bar=None if split=='train' else 0)


def batch_loss(model, batch, device, beta, free_bits):
    enc = batch['enc_input'].permute(2,0,1).to(device)
    dec = batch['dec_input'].T.to(device)
    target = batch['dec_target'].T.to(device)
    mu, logvar, logits = model(enc, dec, batch['bar_pos'].to(device),
        **{name+'_cls':batch[name+'_cls'].T.to(device) for name in ('tempo','density','velocity')},
        padding_mask=batch['enc_padding_mask'].to(device))
    bar_mask = torch.arange(enc.size(2), device=device)[None,:] < batch['enc_n_bars'].to(device)[:,None]
    return model.compute_loss(mu, logvar, beta, free_bits, logits, target, bar_mask=bar_mask)


def beta_cyclical_sched(step, t):
    if t['constant_kl']: return t['kl_max_beta']
    if step < t['no_kl_steps']: return 0.
    progress = ((step-1) % t['kl_cycle_steps']) / t['kl_cycle_steps']
    return t['kl_max_beta'] * min(1., progress*2)


def validate(model, loader, device):
    model.eval()
    values=[]
    with torch.no_grad():
        for batch in loader:
            loss = batch_loss(model, batch, device, 0., 0.)
            values.append([loss['recons_loss'].item(), loss['kldiv_raw'].item()])
    model.train()
    return np.mean(values, axis=0)


def main(path, resume=False):
    with open(path) as f: conf=yaml.safe_load(f)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    t,d=conf['training'],conf['data'];device=t['device']
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime in Colab')
    train_set,val_set=make_dataset(conf,'train'),make_dataset(conf,'val')
    loader=DataLoader(train_set,batch_size=d['batch_size'],shuffle=True,num_workers=d.get('num_workers',0))
    val_loader=DataLoader(val_set,batch_size=d['batch_size'],shuffle=False,num_workers=0)
    model=make_model(conf,train_set.vocab_size).to(device)
    optimizer=optim.Adam(model.parameters(),lr=t['max_lr'])
    os.makedirs(t['ckpt_dir'],exist_ok=True)
    latest=os.path.join(t['ckpt_dir'],'latest.pt');step=0
    if resume:
        state=torch.load(latest,map_location=device,weights_only=False)
        if state['config']['model']!=conf['model']: raise ValueError('Checkpoint architecture differs')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);step=state['step']
    def save():
        # Restart resumes weights/optimizer/step; it reshuffles crops, not exact batch replay.
        tmp=latest+'.tmp'
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'config':conf},tmp)
        os.replace(tmp,latest)
        torch.save(model.state_dict(),os.path.join(t['ckpt_dir'],'model.pt'))
    model.train()
    for epoch in range(t['max_epochs']):
        for batch in loader:
            step+=1; beta=beta_cyclical_sched(step,t)
            if step<t['lr_warmup_steps']: lr=t['max_lr']*step/t['lr_warmup_steps']
            else:
                progress=min(1.,(step-t['lr_warmup_steps'])/t['lr_decay_steps'])
                lr=t['min_lr']+(t['max_lr']-t['min_lr'])*(1+np.cos(np.pi*progress))/2
            for group in optimizer.param_groups: group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            loss=batch_loss(model,batch,device,beta,t['free_bit_lambda'])
            if not torch.isfinite(loss['total_loss']): raise RuntimeError('Non-finite loss')
            loss['total_loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
            if step%t['log_interval']==0:
                print('step',step,'beta',round(beta,4),{k:round(loss[k].item(),5) for k in ('recons_loss','kldiv_raw')},flush=True)
            if step%t['ckpt_interval']==0: save()
            if step%t['val_interval']==0: print('validation RC/KL:',validate(model,val_loader,device),flush=True)
        save()
    print('Saved:',latest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--resume',action='store_true');a=p.parse_args();main(a.config,a.resume)
