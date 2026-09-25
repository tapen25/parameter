import os, pickle, random
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class REMIFullSongTransformerDataset(Dataset):
    def __init__(self, data_dir, vocab_file, model_enc_seqlen=128,
                 model_dec_seqlen=1280, model_max_bars=16, pieces=None,
                 do_augment=False, pad_to_same=True, appoint_st_bar=None):
        self.data_dir = data_dir
        self.event2idx, self.idx2event = pickle_load(vocab_file)
        if sorted(self.event2idx.values()) != list(range(len(self.event2idx))):
            raise ValueError('Vocabulary IDs must be contiguous from zero')
        self.pad_token = len(self.event2idx)
        self.vocab_size = self.pad_token + 1
        self.bar_token = self.event2idx['Bar_None']
        self.eos_token = self.event2idx['EOS_None']
        self.model_enc_seqlen = model_enc_seqlen
        self.model_dec_seqlen = model_dec_seqlen
        self.model_max_bars = model_max_bars
        self.pad_to_same = pad_to_same
        self.appoint_st_bar = appoint_st_bar
        if do_augment:
            raise ValueError('This initial baseline disables pitch augmentation')
        self.pieces = ([os.path.join(data_dir, p) for p in pieces] if pieces is not None
                       else sorted(glob(os.path.join(data_dir, '*.pkl'))))
        if not self.pieces:
            raise ValueError('Empty dataset')

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        path = self.pieces[int(idx)]
        positions, events = pickle_load(path)
        positions = list(positions)
        if positions != [i for i,e in enumerate(events) if e['name'] == 'Bar'] + [len(events)]:
            raise ValueError(f'{path}: incorrect bar boundaries')
        n_total = len(positions)-1
        last_start = max(0, n_total-self.model_max_bars)
        st_bar = random.randint(0, last_start) if self.appoint_st_bar is None else min(self.appoint_st_bar, last_start)
        end_bar = min(n_total, st_bar+self.model_max_bars)
        # Fit whole bars into the decoder window. A single overlong bar is not silently discarded.
        while end_bar > st_bar and positions[end_bar]-positions[st_bar] > self.model_dec_seqlen:
            end_bar -= 1
        if end_bar == st_bar:
            raise ValueError(f'{path}: bar {st_bar} exceeds dec_seqlen; increase the setting')
        start, end = positions[st_bar], positions[end_bar]
        tokens = [self.event2idx[f"{e['name']}_{e['value']}"] for e in events[start:end]]
        n_bars, length = end_bar-st_bar, len(tokens)
        bar_pos = np.array([p-start for p in positions[st_bar:end_bar+1]] +
                           [length]*(self.model_max_bars-n_bars), dtype=np.int64)
        enc = np.full((self.model_max_bars, self.model_enc_seqlen), self.pad_token, dtype=np.int64)
        mask = np.ones_like(enc, dtype=bool)
        enc_lens = np.zeros(self.model_max_bars, dtype=np.int64)
        # Dummy bars retain one visible PAD to avoid all-masked attention NaNs; KL excludes them.
        mask[:, 0] = False
        for b, (st, ed) in enumerate(zip(bar_pos[:n_bars], bar_pos[1:n_bars+1])):
            segment = tokens[st:ed][:self.model_enc_seqlen]
            enc[b, :len(segment)] = segment
            mask[b, :len(segment)] = False
            enc_lens[b] = len(segment)
        size = self.model_dec_seqlen if self.pad_to_same else length
        inp = np.full(size, self.pad_token, dtype=np.int64)
        tgt = np.full(size, self.pad_token, dtype=np.int64)
        inp[:length] = tokens
        # Predict the next Bar at crop boundaries; no fabricated EOS between real bars.
        next_token = self.bar_token if end < len(events) else self.pad_token
        tgt[:length] = tokens[1:] + [next_token]
        result = dict(id=int(idx), piece_id=int(os.path.basename(path).split('.')[0]),
                      st_bar_id=st_bar, bar_pos=bar_pos, enc_input=enc,
                      dec_input=inp, dec_target=tgt, length=length,
                      enc_padding_mask=mask, enc_length=enc_lens, enc_n_bars=n_bars)
        for name in ('tempo', 'density', 'velocity'):
            raw = pickle_load(os.path.join(self.data_dir, 'attr_cls', name, os.path.basename(path)))
            if len(raw) != n_total:
                raise ValueError(f'{path}: {name} labels do not match bar count')
            labels = np.asarray(raw[st_bar:end_bar], dtype=np.int64)
            fill = 8 if name == 'velocity' else 0
            per_bar = np.full(self.model_max_bars, fill, dtype=np.int64)
            per_bar[:n_bars] = labels
            expanded = np.full(size, fill, dtype=np.int64)
            for b, (st, ed) in enumerate(zip(bar_pos[:n_bars], bar_pos[1:n_bars+1])):
                expanded[st:ed] = labels[b]
            result[name+'_cls'] = expanded
            result[name+'_cls_bar'] = per_bar
        return result
