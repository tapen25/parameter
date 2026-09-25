import os, pickle
import numpy as np

# 前処理で表示されたパス。ランタイムを作り直した場合は更新する。
data_dir = '/content/musemorphose_prepared_ag2qlu1e/remi_dataset'

def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def compute_attributes(events, n_bars):
    note_count = np.zeros(n_bars, dtype=int)
    onset_record = np.zeros((n_bars, 16), dtype=int)
    velocity_values = [[] for _ in range(n_bars)]
    tempo_record = np.full(n_bars * 16, np.nan)
    cur_bar, cur_pos = -1, 0

    for ev in events:
        name, value = ev['name'], ev['value']
        if name == 'Bar':
            cur_bar += 1
            cur_pos = 0
            if cur_bar >= n_bars:
                raise ValueError('小節境界とBarイベント数が不一致です')
        elif name == 'Beat':
            cur_pos = int(value)
            if not 0 <= cur_pos < 16:
                raise ValueError('Beatが0〜15の範囲外です')
        elif name in ('Tempo', 'Note_Pitch', 'Note_Velocity'):
            if cur_bar < 0:
                raise ValueError('Barより前に音楽イベントがあります')
            if name == 'Tempo':
                tempo_record[cur_bar * 16 + cur_pos] = float(value)
            elif name == 'Note_Pitch':
                note_count[cur_bar] += 1
                onset_record[cur_bar, cur_pos] = 1
            else:
                velocity_values[cur_bar].append(float(value))

    if cur_bar + 1 != n_bars:
        raise ValueError('小節境界とBarイベント数が不一致です')
    if any(len(v) != count for v, count in zip(velocity_values, note_count)):
        raise ValueError('音符数とVelocity数が不一致。イベントの対応確認が必要です')

    # Tempoは次の指定まで持続。最初の指定以前は不明値NaNのまま。
    current_tempo = np.nan
    for i in range(len(tempo_record)):
        if np.isfinite(tempo_record[i]):
            current_tempo = tempo_record[i]
        tempo_record[i] = current_tempo

    return {
        # 16位置での平均。実時間による加重平均ではない。
        'tempo': tempo_record.reshape(n_bars, 16).mean(axis=1),
        'note_count': note_count,
        'onset_ratio': onset_record.mean(axis=1),
        # 無音小節には平均Velocityを定義できないのでNaNを保存。
        'velocity': np.array([np.mean(v) if v else np.nan for v in velocity_values]),
    }

def main():
    names = ('tempo', 'note_count', 'onset_ratio', 'velocity')
    collected = {name: [] for name in names}
    for name in names:
        os.makedirs(os.path.join(data_dir, 'attr_raw', name), exist_ok=True)
    pieces = sorted(p for p in os.listdir(data_dir) if p.endswith('.pkl'))
    if not pieces:
        raise ValueError('曲ファイルがありません。data_dirを確認してください')
    for i, p in enumerate(pieces):
        bar_pos, events = pickle_load(os.path.join(data_dir, p))
        # 前処理版は末尾にlen(events)という終端境界を含む。
        expected = [j for j,e in enumerate(events) if e['name'] == 'Bar'] + [len(events)]
        if list(bar_pos) != expected:
            raise ValueError(f'{p}: 前処理版の小節境界形式ではありません')
        try:
            attrs = compute_attributes(events, len(bar_pos) - 1)
        except ValueError as exc:
            raise ValueError(f'{p}: {exc}') from exc
        for name, values in attrs.items():
            with open(os.path.join(data_dir, 'attr_raw', name, p), 'wb') as f:
                pickle.dump(values.tolist(), f)
            collected[name].extend(values.tolist())
        if (i+1) % 200 == 0:
            print('処理済み:', i+1)
    for name, values in collected.items():
        values = np.array(values)
        valid = values[np.isfinite(values)]
        print(name, '小節数:', len(values), '不明値:', len(values)-len(valid),
              'min/mean/max:', (valid.min(), valid.mean(), valid.max()) if len(valid) else None)
    print('完了:', os.path.join(data_dir, 'attr_raw'))

if __name__ == '__main__':
    main()
