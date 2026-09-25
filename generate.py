import sys, os, random, time
from copy import deepcopy
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_dir, 'model'))

from dataloader import REMIFullSongTransformerDataset
from model.musemorphose import MuseMorphose

from utils import pickle_load, numpy_to_tensor, tensor_to_numpy
from remi2midi import remi2midi

import torch
import yaml
import numpy as np
from scipy.stats import entropy

config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = config['training']['device']
data_dir = config['data']['data_dir']
vocab_path = config['data']['vocab_path']
data_split = config['data']['test_split']

ckpt_path = sys.argv[2]
out_dir = sys.argv[3]
n_pieces = int(sys.argv[4])
n_samples_per_piece = int(sys.argv[5])

#little helpers
def word2event(word_seq, idx2event):
  return [ idx2event[w] for w in word_seq ]

def get_beat_idx(event):
  return int(event.split('_')[-1])

#sampling utilities
def temperatured_softmax(logits, temperature):
  if temperature <= 0:
    raise ValueError('temperature must be positive')
  scores = np.asarray(logits, dtype=np.float64) / temperature
  probs = np.exp(scores - np.max(scores))
  return probs / probs.sum()

def nucleus(probs, p):
  if not 0 < p <= 1:
    raise ValueError('nucleus_p must be in (0, 1]')
  order = np.argsort(probs)[::-1]
  count = min(len(order), np.searchsorted(np.cumsum(probs[order]), p) + 1)
  candidates = order[:count]
  weights = probs[candidates] / probs[candidates].sum()
  return np.random.choice(candidates, p=weights)

#generation
def get_latent_embedding_fast(model, piece_data, use_sampling=False, sampling_var=0.):
  # reshape
  batch_inp = piece_data['enc_input'].permute(1, 0).long().to(device)
  batch_padding_mask = piece_data['enc_padding_mask'].bool().to(device)

  # get latent conditioning vectors
  with torch.no_grad():
    piece_latents = model.get_sampled_latent(
      batch_inp, padding_mask=batch_padding_mask, 
      use_sampling=use_sampling, sampling_var=sampling_var
    )

  return piece_latents

def generate_on_latent_ctrl_vanilla_truncate(
        model, latents, tempo_cls, density_cls, velocity_cls, event2idx, idx2event, 
        max_events=12800, primer=None,
        max_input_len=1280, truncate_len=512, 
        nucleus_p=0.9, temperature=1.2
      ):
  latent_placeholder = torch.zeros(max_events, 1, latents.size(-1)).to(device)
  tempo_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  density_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  velocity_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  print('[info] tempo:', tempo_cls, 'density:', density_cls, 'velocity:', velocity_cls)

  if primer is not None:
    raise ValueError('This version starts with Bar_None; custom primers are not supported')
  if not 0 < truncate_len < max_input_len <= max_events:
    raise ValueError('Require 0 < truncate_len < max_input_len <= max_events')
  if latents.size(0) == 0:
    raise ValueError('No bars to generate')
  for classes in (tempo_cls, density_cls, velocity_cls):
    if classes.ndim != 1 or len(classes) != latents.size(0):
      raise ValueError('Provide one attribute class per latent bar')
  generated = [event2idx['Bar_None']]

  target_bars, generated_bars = latents.size(0), 0

  steps = 0
  time_st = time.time()
  cur_pos = 0
  failed_cnt = 0

  cur_input_len = len(generated)
  generated_final = deepcopy(generated)
  entropies = []

  while generated_bars < target_bars:
    if len(generated_final) >= max_events:
      raise RuntimeError('Generation reached max_events before completing the requested bars')
    if len(generated) == 1:
      dec_input = numpy_to_tensor([generated], device=device).long()
    else:
      dec_input = numpy_to_tensor([generated], device=device).permute(1, 0).long()

    latent_placeholder[len(generated)-1, 0, :] = latents[ generated_bars ]
    tempo_placeholder[len(generated)-1, 0] = tempo_cls[ generated_bars ]
    density_placeholder[len(generated)-1, 0] = density_cls[ generated_bars ]
    velocity_placeholder[len(generated)-1, 0] = velocity_cls[ generated_bars ]

    dec_seg_emb = latent_placeholder[:len(generated), :]
    dec_tempo_cls = tempo_placeholder[:len(generated), :]
    dec_density_cls = density_placeholder[:len(generated), :]
    dec_velocity_cls = velocity_placeholder[:len(generated), :]

    # sampling
    with torch.no_grad():
      logits = model.generate(dec_input, dec_seg_emb, dec_tempo_cls, dec_density_cls, dec_velocity_cls)
    logits = tensor_to_numpy(logits[0])
    # Exclude padding IDs absent from idx2event and early EOS.
    logits = np.asarray(logits, dtype=np.float64).copy()
    for token_id in range(len(logits)):
      if token_id not in idx2event or idx2event[token_id] == 'PAD_None':
        logits[token_id] = -np.inf
    if generated_bars < target_bars - 1:
      logits[event2idx['EOS_None']] = -np.inf
    probs = temperatured_softmax(logits, temperature)
    word = nucleus(probs, nucleus_p)
    word_event = idx2event[word]

    if 'Beat' in word_event:
      event_pos = get_beat_idx(word_event)
      if not event_pos >= cur_pos:
        failed_cnt += 1
        print ('[info] position not increasing, failed cnt:', failed_cnt)
        if failed_cnt >= 128:
          print ('[FATAL] model stuck, exiting ...')
          raise RuntimeError('Generation stuck at decreasing Beat positions')
        continue
      else:
        cur_pos = event_pos
        failed_cnt = 0

    if 'Bar' in word_event:
      generated_bars += 1
      cur_pos = 0
      print ('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated_final)))
    if word_event == 'PAD_None':
      continue

    if word_event == 'EOS_None':
      generated_bars += 1
      break
    if generated_bars == target_bars:
      break  # Do not append the next bar's delimiter.

    generated.append(word)
    generated_final.append(word)
    entropies.append(entropy(probs))

    cur_input_len += 1
    steps += 1

    assert cur_input_len == len(generated)
    if cur_input_len == max_input_len:
      generated = generated[-truncate_len:]
      latent_placeholder[:len(generated)-1, 0, :] = latent_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0, :].clone()
      tempo_placeholder[:len(generated)-1, 0] = tempo_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0].clone()
      density_placeholder[:len(generated)-1, 0] = density_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0].clone()
      velocity_placeholder[:len(generated)-1, 0] = velocity_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0].clone()

      print ('[info] reset context length: cur_len: {}, accumulated_len: {}, truncate_range: {} ~ {}'.format(
        cur_input_len, len(generated_final), cur_input_len-truncate_len, cur_input_len-1
      ))
      cur_input_len = len(generated)

  assert generated_bars == target_bars
  print ('-- generated events:', len(generated_final))
  print ('-- time elapsed: {:.2f} secs'.format(time.time() - time_st))
  return generated_final, time.time() - time_st, np.array(entropies)


# change attribute classes
def specified_classes(piece_data, name, n_bars):
  # Optional YAML generate.controls.<name>: scalar class or a list of n_bars classes.
  original = tensor_to_numpy(piece_data[name + '_cls_bar'])[:n_bars].astype(np.int64)
  values = config['generate'].get('controls', {}).get(name, original)
  values = np.asarray(values)
  if values.ndim == 0:
    values = np.full(n_bars, values.item())
  max_id = config['model']['n_' + name + '_cls'] - 1
  if values.shape != (n_bars,) or not np.issubdtype(values.dtype, np.integer):
    raise ValueError(name + ': provide an integer class or a per-bar integer list')
  if np.any(values < 0) or np.any(values > max_id):
    raise ValueError(name + ': class ID out of range')
  return torch.as_tensor(values, dtype=torch.long, device=device)


if __name__ == "__main__":
  dset = REMIFullSongTransformerDataset(
    data_dir, vocab_path, 
    do_augment=False,
    model_enc_seqlen=config['data']['enc_seqlen'], 
    model_dec_seqlen=config['generate']['dec_seqlen'],
    model_max_bars=config['generate']['max_bars'],
    pieces=pickle_load(data_split),
    pad_to_same=False
  )
  pieces = random.sample(range(len(dset)), n_pieces)
  print ('[sampled pieces]', pieces)
  
  mconf = config['model']
  model = MuseMorphose(
    mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
    mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
    mconf['d_latent'], mconf['d_embed'], dset.vocab_size,
    d_tempo_emb=mconf['d_tempo_emb'], n_tempo_cls=mconf['n_tempo_cls'],
    d_density_emb=mconf['d_density_emb'], n_density_cls=mconf['n_density_cls'],
    d_velocity_emb=mconf['d_velocity_emb'], n_velocity_cls=mconf['n_velocity_cls'],
    cond_mode=mconf['cond_mode']
  ).to(device)
  model.eval()
  model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  times = []
  piece_entropies = []
  for p in pieces:
    # fetch test sample
    p_data = dset[p]
    p_id = p_data['piece_id']
    p_bar_id = p_data['st_bar_id']
    p_data['enc_input'] = p_data['enc_input'][ : p_data['enc_n_bars'] ]
    p_data['enc_padding_mask'] = p_data['enc_padding_mask'][ : p_data['enc_n_bars'] ]


    orig_song = p_data['dec_input'].tolist()[:p_data['length']]
    orig_song = word2event(orig_song, dset.idx2event)
    orig_out_file = os.path.join(out_dir, 'id{}_bar{}_orig'.format(
        p, p_bar_id
    ))
    print ('[info] writing to ...', orig_out_file)
    # output reference song's MIDI
    _, orig_tempo = remi2midi(orig_song, orig_out_file + '.mid', return_first_tempo=True, enforce_tempo=False)

    # save metadata of reference song (events & attr classes)
    print (*orig_song, sep='\n', file=open(orig_out_file + '.txt', 'a'))
    np.save(orig_out_file + '-VELOCITYCLS.npy', p_data['velocity_cls_bar'])
    np.save(orig_out_file + '-DENSITYCLS.npy', p_data['density_cls_bar'])
    np.save(orig_out_file + '-TEMPOCLS.npy', p_data['tempo_cls_bar'])


    for k in p_data.keys():
      if not torch.is_tensor(p_data[k]):
        p_data[k] = numpy_to_tensor(p_data[k], device=device)
      else:
        p_data[k] = p_data[k].to(device)

    p_latents = get_latent_embedding_fast(
                  model, p_data, 
                  use_sampling=config['generate']['use_latent_sampling'],
                  sampling_var=config['generate']['latent_sampling_var']
                )

    for samp in range(n_samples_per_piece):
      p_tempo_cls = specified_classes(p_data, 'tempo', len(p_latents))
      p_density_cls = specified_classes(p_data, 'density', len(p_latents))
      p_velocity_cls = specified_classes(p_data, 'velocity', len(p_latents))

      print ('[info] piece: {}, bar: {}'.format(p_id, p_bar_id))
      out_file = os.path.join(out_dir, f'id{p}_bar{p_bar_id}_sample{samp+1:02d}')
      print ('[info] writing to ...', out_file)
      if os.path.exists(out_file + '.txt'):
        print ('[info] file exists, skipping ...')
        continue

      # print (p_velocity_cls, p_density_cls)

      # generate
      song, t_sec, entropies = generate_on_latent_ctrl_vanilla_truncate(
                                  model, p_latents, p_tempo_cls, p_density_cls, p_velocity_cls, dset.event2idx, dset.idx2event,
                                  max_input_len=config['generate']['max_input_dec_seqlen'], 
                                  truncate_len=min(512, config['generate']['max_input_dec_seqlen'] - 32), 
                                  nucleus_p=config['generate']['nucleus_p'], 
                                  temperature=config['generate']['temperature'],
                                  
                                )
      times.append(t_sec)

      song = word2event(song, dset.idx2event)
      print (*song, sep='\n', file=open(out_file + '.txt', 'a'))
      remi2midi(song, out_file + '.mid', enforce_tempo=False)

      # save metadata of the generation
      np.save(out_file + '-VELOCITYCLS.npy', tensor_to_numpy(p_velocity_cls))
      np.save(out_file + '-DENSITYCLS.npy', tensor_to_numpy(p_density_cls))
      np.save(out_file + '-TEMPOCLS.npy', tensor_to_numpy(p_tempo_cls))
      print ('[info] piece entropy: {:.4f} (+/- {:.4f})'.format(
        entropies.mean(), entropies.std()
      ))
      piece_entropies.append(entropies.mean())

  print ('[time stats] {} songs, generation time: {:.2f} secs (+/- {:.2f})'.format(
    n_pieces * n_samples_per_piece, np.mean(times), np.std(times)
  ))
  print ('[entropy] {:.4f} (+/- {:.4f})'.format(
    np.mean(piece_entropies), np.std(piece_entropies)
  ))
