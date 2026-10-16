#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""音楽ファイル詳細分析スクリプト（AIに「耳」をつける）

使い方:
  python analyze_audio.py <音声ファイル>                    # 基本（曲の全長）
  python analyze_audio.py <音声ファイル> --deep             # 音色・和声追加
  python analyze_audio.py <音声ファイル> --structure        # 構造（区間ごとの中身つき）
  python analyze_audio.py <音声ファイル> --rhythm           # リズム詳細
  python analyze_audio.py <音声ファイル> --bands            # 周波数帯別（元のサンプルレートで）
  python analyze_audio.py <音声ファイル> --hpss             # 和声/打音分離
  python analyze_audio.py <音声ファイル> --pitch            # F0追跡
  python analyze_audio.py <音声ファイル> --tags             # PANNsタグ（全長を10秒ごとに）
  python analyze_audio.py <音声ファイル> --game             # ゲームBGM用チェック（無音・DC・ループ候補）
  python analyze_audio.py <音声ファイル> --summary          # 日本語の短いまとめだけ出す
  python analyze_audio.py <音声ファイル> --whisper [--lang ja]   # 歌詞文字起こし
  python analyze_audio.py <音声ファイル> --all              # 全部（Whisper large と Demucs 以外）
  python analyze_audio.py <音声ファイル> --offset 60 --duration 30   # 一部だけ

2026/10 改訂（ニコイ）:
  - 既定で曲の全長を聴く（前は最初の30秒だけ）。一部だけなら --duration
  - 周波数帯・ピーク・音割れは元のサンプルレート（ステレオ）で測る（前は22kHzモノラルで、11kHz以上が消えていた）
  - ピークは sample peak と true peak（4倍オーバーサンプル）を分け、音割れしたサンプル数を数える
  - 区間の数を曲の長さに合わせて決め、拍にそろえ、区間ごとの音量・打楽器・低音・明るさを出す
  - PANNs タグを全長で（10秒ごと）聴き、時間ごとのタグも出す
  - Demucs は今のPythonで呼び、分離した楽器の大きさを全長で測る。Whisper の言語は自動判定（--lang で指定）
"""
import sys
import json
import argparse
import contextlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import librosa.feature

sys.stdout.reconfigure(encoding='utf-8')

ANALYSIS_SR = 22050  # テンポ・クロマなど重い特徴量用（元の音は別に保持して帯域・ピークに使う）


class NumpyJSONEncoder(json.JSONEncoder):
    """numpy型を変換するエンコーダ"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def db(x):
    return 20 * np.log10(np.maximum(x, 1e-10))


class Audio:
    """一度だけ読み込んで、各分析で使い回す"""
    def __init__(self, path, offset, duration):
        y, sr = librosa.load(path, sr=None, mono=False, offset=offset, duration=duration)
        if y.ndim == 1:
            y = np.stack([y, y])
            self.is_mono_source = True
        else:
            self.is_mono_source = False
        self.path = path
        self.stereo = y            # (2, n) 元のサンプルレート
        self.sr_native = sr
        self.mono_native = y.mean(axis=0)
        self.y = librosa.resample(self.mono_native, orig_sr=sr, target_sr=ANALYSIS_SR) if sr != ANALYSIS_SR else self.mono_native
        self.sr = ANALYSIS_SR
        self.offset = offset
        self._hpss = None
        self._beats = None

    @property
    def duration(self):
        return self.stereo.shape[1] / self.sr_native

    def hpss(self):
        if self._hpss is None:
            self._hpss = librosa.effects.hpss(self.y)
        return self._hpss

    def beats(self):
        if self._beats is None:
            onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
            tempo = float(tempo) if np.isscalar(tempo) else float(np.asarray(tempo).ravel()[0])
            beat_times = librosa.frames_to_time(beats, sr=self.sr)
            self.tempo_doubled = False
            if 0 < tempo < 80 and len(beat_times) >= 2:
                # 和音を2拍ごとに弾き直す曲などで、テンポを半分に読むことがある。80未満なら倍にして、拍のあいだに拍を足す
                tempo *= 2
                mids = (beat_times[:-1] + beat_times[1:]) / 2
                beat_times = np.sort(np.concatenate([beat_times, mids, [beat_times[-1] + (beat_times[-1] - beat_times[-2]) / 2]]))
                beat_times = beat_times[beat_times < self.duration]
                self.tempo_doubled = True
            expected = self.duration * tempo / 60 if tempo > 0 else 0
            self.beats_from_grid = False
            if tempo > 0 and len(beat_times) < 0.5 * expected:
                # アタックの少ない曲（オルガン・パッド中心など）は拍が拾えないことがある。
                # そのときはテンポから拍の目盛りを作り、音の立ち上がりがいちばん重なる位置にずらす
                period = 60.0 / tempo
                t_env = librosa.frames_to_time(np.arange(len(onset_env)), sr=self.sr)
                best_phase, best_score = 0.0, -1.0
                for phase in np.linspace(0, period, 32, endpoint=False):
                    grid = np.arange(phase, self.duration, period)
                    score = float(np.interp(grid, t_env, onset_env).sum())
                    if score > best_score:
                        best_phase, best_score = phase, score
                beat_times = np.arange(best_phase, self.duration, period)
                self.beats_from_grid = True
            self._beats = (tempo, beat_times)
        return self._beats


def estimate_key(chroma_mean):
    best = (None, None, -np.inf)
    for i in range(12):
        rotated = np.roll(chroma_mean, -i)
        maj = np.corrcoef(rotated, MAJOR_PROFILE)[0, 1]
        mnr = np.corrcoef(rotated, MINOR_PROFILE)[0, 1]
        if maj > best[2]:
            best = (NOTES[i], 'major', maj)
        if mnr > best[2]:
            best = (NOTES[i], 'minor', mnr)
    return best[0], best[1], float(best[2])


def peak_analysis(a):
    """ピークと音割れ：元のサンプルレート・左右別々で測る"""
    from scipy.signal import resample_poly
    x = a.stereo
    sample_peak = float(np.abs(x).max())
    clipped = int((np.abs(x) >= 0.999).sum())
    # true peak：4倍オーバーサンプルして、サンプルとサンプルの間の山も見る（長い曲は区切って計算）
    tp = 0.0
    step = a.sr_native * 30
    for ch in x:
        for i in range(0, len(ch), step):
            seg = ch[max(0, i - 64): i + step + 64]
            tp = max(tp, float(np.abs(resample_poly(seg, 4, 1)).max()))
    return {
        'sample_peak_dbfs': round(float(db(sample_peak)), 2),
        'true_peak_dbtp': round(float(db(tp)), 2),
        'clipped_samples': clipped,
        'clipped_ratio': round(clipped / x.size, 7),
        'note': 'true peak が 0 を超えると、再生機器やmp3変換で割れることがある。clipped_samples は ±0.999 以上のサンプル数',
    }


def basic_analysis(a):
    """基本：テンポ・キー・スペクトル・音量"""
    y, sr = a.y, a.sr
    tempo, beat_times = a.beats()
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_note, key_mode, key_conf = estimate_key(chroma.mean(axis=1))

    yn, srn = a.mono_native, a.sr_native
    spec_centroid = float(librosa.feature.spectral_centroid(y=yn, sr=srn).mean())
    spec_rolloff = float(librosa.feature.spectral_rolloff(y=yn, sr=srn).mean())
    spec_bandwidth = float(librosa.feature.spectral_bandwidth(y=yn, sr=srn).mean())
    zcr = float(librosa.feature.zero_crossing_rate(yn).mean())
    rms = float(librosa.feature.rms(y=yn).mean())
    peaks = peak_analysis(a)
    rms_db = float(db(rms))

    return {
        'tempo_bpm': round(tempo, 2),
        'beats_count': int(len(beat_times)),
        'beats_from_tempo_grid': bool(getattr(a, 'beats_from_grid', False)),
        'tempo_doubled_from_half': bool(getattr(a, 'tempo_doubled', False)),
        'key': {'note': key_note, 'mode': key_mode, 'confidence': round(key_conf, 3)},
        'spectral': {
            'centroid_hz': round(spec_centroid, 1),
            'rolloff_hz': round(spec_rolloff, 1),
            'bandwidth_hz': round(spec_bandwidth, 1),
            'zero_crossing_rate': round(zcr, 4),
            'note': f'元のサンプルレート {srn}Hz で測定',
        },
        'loudness': {
            'peak_db': peaks['sample_peak_dbfs'],
            'true_peak_dbtp': peaks['true_peak_dbtp'],
            'clipped_samples': peaks['clipped_samples'],
            'rms_db': round(rms_db, 2),
            'dynamic_range_db': round(peaks['sample_peak_dbfs'] - rms_db, 2),
            'note': peaks['note'],
        },
    }


def deep_analysis(a):
    """音色・和声詳細：MFCC・トンネッツ・スペクトラルコントラスト"""
    y, sr = a.y, a.sr
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=a.mono_native, sr=a.sr_native)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)

    return {
        'mfcc': {
            'mean': mfcc.mean(axis=1).round(2).tolist(),
            'std': mfcc.std(axis=1).round(2).tolist(),
            'note': 'MFCC 13次元の音色指紋。1次元目は音量、2-13は音色特性',
        },
        'tonnetz': {
            'mean': tonnetz.mean(axis=1).round(3).tolist(),
            'note': '和声的特徴。[5度,短3度,長3度] x [円x,円y] の6次元',
        },
        'spectral_contrast': {
            'mean': contrast.mean(axis=1).round(2).tolist(),
            'note': '7帯域それぞれの「ピーク-谷」のdB差。値が高いほどメリハリある（元のサンプルレートで測定）',
        },
        'chroma_distribution': {
            'notes': NOTES,
            'normalized': (chroma / chroma.max()).round(3).tolist(),
            'note': '各音の出現強度。最大値を1.0に正規化',
        },
    }


def _band_share(S, freqs, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    return S[mask].sum(axis=0)


def structure_analysis(a):
    """構造：区間の数は曲の長さから決め、境界を拍にそろえ、区間ごとの中身を出す"""
    y, sr = a.y, a.sr
    hop = 2048
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    feats = np.vstack([librosa.util.normalize(chroma, axis=0), librosa.util.normalize(mfcc, axis=1)])
    dur = a.duration
    k = int(np.clip(round(dur / 20), 2, 12))   # だいたい20秒に1区間。短い曲は2つまで
    bounds = librosa.segment.agglomerative(feats, k=k)
    bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=hop)

    tempo, beat_times = a.beats()
    snapped = []
    for t in bound_times:
        if t <= 0.01 or len(beat_times) == 0:
            snapped.append(0.0)
        else:
            snapped.append(float(beat_times[np.argmin(np.abs(beat_times - t))]))
    edges = sorted(set([0.0] + [t for t in snapped if 0 < t < dur - 1.0])) + [dur]

    # 区間ごとの中身
    y_h, y_p = a.hpss()
    S = np.abs(librosa.stft(a.mono_native, n_fft=4096, hop_length=1024)) ** 2
    freqs = librosa.fft_frequencies(sr=a.sr_native, n_fft=4096)
    t_S = librosa.frames_to_time(np.arange(S.shape[1]), sr=a.sr_native, hop_length=1024)
    low = _band_share(S, freqs, 20, 250)
    total = S.sum(axis=0) + 1e-12
    centroid = (freqs[:, None] * S).sum(axis=0) / total
    rms_all = librosa.feature.rms(y=a.mono_native, frame_length=4096, hop_length=1024).flatten()[: S.shape[1]]

    sections = []
    for i in range(len(edges) - 1):
        s, e = edges[i], edges[i + 1]
        m = (t_S >= s) & (t_S < e)
        hs, he = int(s * sr), int(e * sr)
        ph, pp = float(np.sum(y_h[hs:he] ** 2)), float(np.sum(y_p[hs:he] ** 2))
        sections.append({
            'start_sec': round(s, 2), 'end_sec': round(e, 2), 'length_sec': round(e - s, 2),
            'mean_db': round(float(db(np.sqrt(np.mean(rms_all[m] ** 2)))) if m.any() else -100, 1),
            'percussive_ratio': round(pp / (ph + pp + 1e-12), 3),
            'low_end_ratio': round(float(low[m].sum() / total[m].sum()), 3) if m.any() else 0,
            'brightness_hz': round(float(np.median(centroid[m])), 0) if m.any() else 0,
        })
    if sections:
        loud = max(sections, key=lambda x: x['mean_db'])
        loud_db = loud['mean_db']
        for sec in sections:
            gap = loud_db - sec['mean_db']
            sec['label'] = 'いちばん盛り上がる' if sec is loud else ('静か' if gap > 9 else ('中くらい' if gap > 4 else '盛り上がり'))

    # 打楽器が入ってくる時刻（打音エネルギーが曲中の最大の30%を超えた最初の時刻）
    perc_rms = librosa.feature.rms(y=y_p, hop_length=hop).flatten()
    perc_t = librosa.frames_to_time(np.arange(len(perc_rms)), sr=sr, hop_length=hop)
    smooth = np.convolve(perc_rms, np.ones(8) / 8, mode='same')
    over = np.nonzero(smooth > 0.3 * smooth.max())[0]
    perc_share = float(np.sum(y_p ** 2) / (np.sum(y_h ** 2) + np.sum(y_p ** 2) + 1e-12))
    # 打楽器が全体の5%未満の曲では「入ってくる時刻」は意味がないので出さない
    drums_in = round(float(perc_t[over[0]]), 2) if len(over) and perc_share >= 0.05 else None

    energy = librosa.feature.rms(y=y, hop_length=hop).flatten()
    e_t = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop)
    step = max(1, int(round(1.0 / (hop / sr))))   # だいたい1秒ごとに間引く
    return {
        'section_boundaries_sec': [round(t, 2) for t in edges[:-1]],
        'sections_count': len(sections),
        'sections': sections,
        'percussion_enters_sec': drums_in,
        'energy_curve_db': {
            'times_sec': [round(float(t), 1) for t in e_t[::step]],
            'values_db': [round(float(v), 1) for v in db(energy)[::step]],
            'note': '約1秒ごとのRMSラウドネス',
        },
        'note': f'区間の数は曲の長さから自動（k={k}）。境界は一番近い拍にそろえた。label は一番大きい区間との差で付けた目安',
    }


def rhythm_analysis(a):
    """リズム詳細：オンセット・テンポの安定"""
    y, sr = a.y, a.sr
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).round(3).tolist()
    tempo_per_frame = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
    return {
        'onset_count': len(onset_times),
        'onset_density_per_sec': round(len(onset_times) / (len(y) / sr), 2),
        'first_10_onsets_sec': onset_times[:10],
        'tempo_stability_std': round(float(np.std(tempo_per_frame)), 2),
        'tempo_stability_note': '低いほど安定。10以下なら安定、30以上は揺れ大',
    }


def bands_analysis(a):
    """周波数帯別エネルギー：元のサンプルレートで測る（前は22kHzに落としていて11kHz以上が消えていた）"""
    S = np.abs(librosa.stft(a.mono_native, n_fft=4096))
    freqs = librosa.fft_frequencies(sr=a.sr_native, n_fft=4096)
    nyq = a.sr_native / 2
    bands = {
        'sub_bass_20_60hz': (20, 60),
        'bass_60_250hz': (60, 250),
        'low_mid_250_500hz': (250, 500),
        'mid_500_2000hz': (500, 2000),
        'high_mid_2000_4000hz': (2000, 4000),
        'presence_4000_6000hz': (4000, 6000),
        'brilliance_6000_20000hz': (6000, 20000),
    }
    total = S.sum()
    result = {}
    for name, (lo, hi) in bands.items():
        ratio = float(S[(freqs >= lo) & (freqs < hi)].sum() / total) if total > 0 else 0
        result[name] = {'energy_ratio': round(ratio, 4), 'percent': round(ratio * 100, 1)}
    air = float(S[(freqs >= 10000) & (freqs < 20000)].sum() / total) if total > 0 else 0
    result['air_10000_20000hz_within_brilliance'] = {'energy_ratio': round(air, 4), 'percent': round(air * 100, 1)}
    result['note'] = f'元のサンプルレート {a.sr_native}Hz（{int(nyq)}Hzまで）で測定。air はシンバルや空気感の帯域'
    return result


def hpss_analysis(a):
    """和声/打音分離"""
    y_harm, y_perc = a.hpss()
    h_energy = float(np.sum(y_harm ** 2))
    p_energy = float(np.sum(y_perc ** 2))
    total = h_energy + p_energy
    return {
        'harmonic_ratio': round(h_energy / total, 3),
        'percussive_ratio': round(p_energy / total, 3),
        'note': '和声成分（メロディ・コード）と打音成分（ドラム）の比率',
    }


def pitch_analysis(a):
    """F0追跡（pyin）"""
    f0, voiced_flag, voiced_probs = librosa.pyin(
        a.y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=a.sr
    )
    voiced_f0 = f0[~np.isnan(f0)]
    if len(voiced_f0) == 0:
        return {'note': 'F0検出失敗（モノフォニック音源向け）'}
    pitches_midi = librosa.hz_to_midi(voiced_f0)
    return {
        'f0_count': int(len(voiced_f0)),
        'f0_mean_hz': round(float(np.mean(voiced_f0)), 2),
        'f0_median_hz': round(float(np.median(voiced_f0)), 2),
        'f0_min_hz': round(float(np.min(voiced_f0)), 2),
        'f0_max_hz': round(float(np.max(voiced_f0)), 2),
        'pitch_range_semitones': round(float(np.max(pitches_midi) - np.min(pitches_midi)), 1),
        'voicing_ratio': round(float(np.sum(~np.isnan(f0)) / len(f0)), 3),
        'first_30_notes': [librosa.midi_to_note(int(round(m))) for m in pitches_midi[::10]][:30],
        'note': 'pyinによるモノフォニック音高追跡。ポリフォニックには弱い',
    }


def lufs_analysis(a):
    """LUFS ラウドネス測定（プロマスタリング基準）"""
    import pyloudnorm as pyln
    y_t = a.stereo.T
    sr = a.sr_native
    meter = pyln.Meter(sr)
    integrated = meter.integrated_loudness(y_t)
    block = int(sr * 3.0)
    stl = []
    for i in range(0, len(y_t) - block, block // 2):
        try:
            v = meter.integrated_loudness(y_t[i:i + block])
            if np.isfinite(v):
                stl.append(v)
        except Exception:
            pass
    targets = {'Spotify': -14.0, 'YouTube': -14.0, 'Apple Music': -16.0, 'Tidal': -14.0, 'CD/Streaming Pro': -10.0}
    return {
        'integrated_lufs': round(float(integrated), 2),
        'short_term_lufs_min': round(float(min(stl)), 2) if stl else None,
        'short_term_lufs_max': round(float(max(stl)), 2) if stl else None,
        'short_term_lufs_range': round(float(max(stl) - min(stl)), 2) if stl else None,
        'headroom_to_targets_db': {k: round(integrated - v, 2) for k, v in targets.items()},
        'note': '正の値は基準より大きい、負は静か。Spotify基準は-14 LUFS',
    }


def stereo_analysis(a):
    """ステレオイメージ分析（L/R・位相・幅）"""
    if a.is_mono_source:
        return {'note': 'モノラル音源なのでステレオ分析不可'}
    L, R = a.stereo[0], a.stereo[1]
    mid, side = (L + R) / 2, (L - R) / 2
    mid_rms, side_rms = float(np.sqrt(np.mean(mid ** 2))), float(np.sqrt(np.mean(side ** 2)))
    L_rms, R_rms = float(np.sqrt(np.mean(L ** 2))), float(np.sqrt(np.mean(R ** 2)))
    return {
        'L_R_correlation': round(float(np.corrcoef(L, R)[0, 1]), 3),
        'L_R_correlation_note': '1.0=モノ、0=完全ステレオ、-1=位相反転（要注意）',
        'stereo_width_ratio': round(side_rms / (mid_rms + 1e-10), 3),
        'stereo_width_note': '0=モノ、1.0前後=広い、大きすぎはリスニング疲れる',
        'lr_balance_db': round(float(db(R_rms) - db(L_rms)), 2),
        'lr_balance_note': '正=右寄り、負=左寄り、0付近=センター。±0.5dB以内が理想',
        'mid_energy_db': round(float(db(mid_rms)), 2),
        'side_energy_db': round(float(db(side_rms)), 2),
    }


def demucs_separate(path, out_dir=None):
    """Demucs で楽器分離（vocal/drums/bass/other）"""
    import subprocess
    import os
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(path) or '.', 'demucs_out')
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, '-m', 'demucs', '-o', out_dir, path]   # いま動いているPythonで呼ぶ
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {'error': result.stderr[-500:], 'note': 'Demucs実行失敗'}
    name = os.path.splitext(os.path.basename(path))[0]
    sep_dir = os.path.join(out_dir, 'htdemucs', name)
    stems = {}
    if os.path.isdir(sep_dir):
        for f in os.listdir(sep_dir):
            stems[os.path.splitext(f)[0]] = os.path.join(sep_dir, f)
    stem_energy = {}
    for stem, path_stem in stems.items():
        y_s, _ = librosa.load(path_stem, sr=ANALYSIS_SR, mono=True)   # 全長で測る
        stem_energy[stem] = {'file': path_stem, 'rms_db': round(float(db(np.sqrt(np.mean(y_s ** 2)))), 2)}
    return {
        'stems_dir': sep_dir,
        'stems': stem_energy,
        'note': 'vocals/drums/bass/other に分離。stem毎にRMSエネルギー（全長）',
    }


GENRE_KW = ['music', 'rock', 'pop', 'jazz', 'classical', 'hip hop', 'electronic', 'metal', 'punk', 'folk',
            'country', 'blues', 'funk', 'disco', 'reggae', 'soul', 'r&b', 'house', 'techno', 'trance',
            'dubstep', 'dance', 'ambient', 'soundtrack', 'orchestra']
INSTRUMENT_KW = ['guitar', 'drum', 'bass', 'piano', 'violin', 'cello', 'flute', 'saxophone', 'trumpet',
                 'synthesizer', 'organ', 'harp', 'banjo', 'percussion', 'cymbal', 'snare', 'hi-hat',
                 'sampler', 'string', 'wind', 'brass', 'keyboard']
VOCAL_KW = ['singing', 'speech', 'vocal', 'shout', 'scream', 'rap', 'choir', 'whisper']
MOOD_KW = ['happy', 'sad', 'aggressive', 'chill', 'exciting', 'scary', 'tense', 'funny', 'cacophony']


def panns_tags(a, top_n=40, window_sec=10.0):
    """PANNs 音響タグ付け：曲の全長を10秒ごとに聴いて、全体と時間ごとのタグを出す"""
    from panns_inference import AudioTagging, labels as panns_labels
    y = librosa.resample(a.mono_native, orig_sr=a.sr_native, target_sr=32000)
    win = int(window_sec * 32000)
    n = max(1, int(np.ceil(len(y) / win)))
    batch = np.zeros((n, win), dtype=np.float32)
    for i in range(n):
        seg = y[i * win:(i + 1) * win]
        batch[i, :len(seg)] = seg
    # PANNs は読み込み時に標準出力へ文字を出すので、JSON が壊れないよう標準エラーへ逃がす
    with contextlib.redirect_stdout(sys.stderr):
        at = AudioTagging(checkpoint_path=None, device='cpu')
        clip, _ = at.inference(batch)
    clip = np.asarray(clip)
    overall = clip.max(axis=0)   # どこかで鳴っていればタグが立つように、窓ごとの最大
    top_idx = np.argsort(overall)[::-1][:top_n]
    tags = [{'tag': panns_labels[i], 'score': round(float(overall[i]), 4)} for i in top_idx]

    def collect(keywords):
        return [t for t in tags if any(k in t['tag'].lower() for k in keywords)][:10]

    timeline = []
    real_len = [min(win, len(y) - i * win) / 32000 for i in range(n)]
    if n > 1 and real_len[-1] < 3.0:
        # 最後の窓が3秒未満（ほとんど無音で埋めた窓）は、時間ごとのタグにも全体のタグにも使わない
        n -= 1
        clip = clip[:n]
        overall = clip.max(axis=0)
        top_idx = np.argsort(overall)[::-1][:top_n]
        tags = [{'tag': panns_labels[i], 'score': round(float(overall[i]), 4)} for i in top_idx]
    for i in range(n):
        idx = np.argsort(clip[i])[::-1]
        picks = [panns_labels[j] for j in idx if panns_labels[j] not in ('Music',)][:3]
        timeline.append({'start_sec': round(a.offset + i * window_sec, 1), 'top': picks})
    return {
        'top_tags': tags[:15],
        'genres': collect(GENRE_KW),
        'instruments': collect(INSTRUMENT_KW),
        'vocals': collect(VOCAL_KW),
        'moods': collect(MOOD_KW),
        'timeline': timeline,
        'note': 'PANNs Cnn14 (AudioSet 527 tags)。全長を10秒ごとに聴き、スコアは窓ごとの最大。0-1で相対的に解釈',
    }


def game_checks(a):
    """ゲームBGM用：頭とお尻の無音・DCずれ・ループの候補（区間の境界×拍）"""
    x = a.mono_native
    sr = a.sr_native
    hop = 1024
    peaks = np.array([np.abs(x[i:i + hop]).max() for i in range(0, len(x) - hop, hop)])
    loud = np.nonzero(db(peaks) > -50)[0]
    head = round(loud[0] * hop / sr, 3) if len(loud) else None
    tail = round((len(peaks) - 1 - loud[-1]) * hop / sr, 3) if len(loud) else None
    dc = [round(float(ch.mean()), 5) for ch in a.stereo]

    # ループ候補：拍の上で、似た手ざわりの組を探す（区間の境界を優先）
    tempo, beat_times = a.beats()
    cands = []
    if len(beat_times) > 16:
        hop2 = 512
        chroma = librosa.feature.chroma_cqt(y=a.y, sr=a.sr, hop_length=hop2)
        mfcc = librosa.feature.mfcc(y=a.y, sr=a.sr, n_mfcc=13, hop_length=hop2)
        F = np.vstack([librosa.util.normalize(chroma, axis=0), librosa.util.normalize(mfcc, axis=1) * 0.5]).T
        F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
        L = int(4.0 * a.sr / hop2)
        bf = librosa.time_to_frames(beat_times, sr=a.sr, hop_length=hop2)
        bf = bf[bf + L < len(F)]
        starts = bf[(bf * hop2 / a.sr) < a.duration * 0.4]
        ends = bf[(bf * hop2 / a.sr) > a.duration * 0.5]
        for s in starts[:: max(1, len(starts) // 40)]:
            for e in ends[:: max(1, len(ends) // 80)]:
                if (e - s) * hop2 / a.sr < 8:
                    continue
                sim = float(np.mean(np.sum(F[s:s + L] * F[e:e + L], axis=1)))
                cands.append((sim, s * hop2 / a.sr, e * hop2 / a.sr))
        cands.sort(reverse=True)
    picked = []
    for sim, s, e in cands:
        if all(abs(s - p['start_sec']) > 2 or abs(e - p['end_sec']) > 2 for p in picked):
            picked.append({'start_sec': round(s, 2), 'end_sec': round(e, 2), 'length_sec': round(e - s, 2), 'similarity': round(sim, 3)})
        if len(picked) >= 5:
            break
    return {
        'leading_silence_sec': head,
        'trailing_silence_sec': tail,
        'dc_offset_LR': dc,
        'loop_candidates': picked,
        'note': 'loop_candidates は拍の上で、ループの始まりと終わりの手ざわりが似ている組（1に近いほど似ている）。loop.py の --start/--end に渡せる。最後に決めるのは耳',
    }


THIRD_OCT_CENTERS = [25 * 2 ** (k / 3) for k in range(0, 30)]   # 25Hz〜約16kHz（1/3オクターブ）
REGIONS = [  # 話すときの名前（帯域の呼び名は目安）
    ('超低域（体で感じる）', 20, 60),
    ('低音の芯', 60, 150),
    ('こもり', 200, 500),
    ('鼻づまり感', 700, 1500),
    ('耳に痛いところ', 2000, 5000),
    ('歯擦音・シンバルの刺さり', 5000, 9000),
    ('空気感', 10000, 20000),
]


def ltas_third_octave(a):
    """曲全体の平均的な音の形（1/3オクターブごとのdB）"""
    from scipy.signal import welch
    f, P = welch(a.mono_native, fs=a.sr_native, nperseg=8192)
    nyq = a.sr_native / 2
    centers, vals = [], []
    for fc in THIRD_OCT_CENTERS + [20000]:
        lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
        if hi > nyq:
            break
        m = (f >= lo) & (f < hi)
        if m.any():
            centers.append(round(fc, 1))
            vals.append(float(10 * np.log10(P[m].mean() + 1e-20)))
    return np.array(centers), np.array(vals)


def region_level(centers, vals, lo, hi):
    m = (centers >= lo) & (centers < hi)
    return float(10 * np.log10(np.mean(10 ** (vals[m] / 10)))) if m.any() else None


def quick_metrics(a):
    """くらべる用の数字（重い分析はしない）"""
    import pyloudnorm as pyln
    b = basic_analysis(a)
    lufs = float(pyln.Meter(a.sr_native).integrated_loudness(a.stereo.T))
    y_h, y_p = a.hpss()
    ph, pp = float(np.sum(y_h ** 2)), float(np.sum(y_p ** 2))
    st = stereo_analysis(a)
    rms = float(np.sqrt(np.mean(a.mono_native ** 2)))
    return {
        'duration_sec': round(a.duration, 2),
        'tempo_bpm': b['tempo_bpm'],
        'key': f"{b['key']['note']} {b['key']['mode']}",
        'integrated_lufs': round(lufs, 2),
        'true_peak_dbtp': b['loudness']['true_peak_dbtp'],
        'clipped_samples': b['loudness']['clipped_samples'],
        'crest_factor_db': round(float(db(np.abs(a.mono_native).max()) - db(rms)), 2),
        'percussive_ratio': round(pp / (ph + pp + 1e-12), 3),
        'brightness_centroid_hz': b['spectral']['centroid_hz'],
        'stereo_correlation': st.get('L_R_correlation'),
        'stereo_width': st.get('stereo_width_ratio'),
    }


def compare_analysis(a, b, image_path=None):
    """A と B をくらべる。音の形は「音の大きさ（LUFS）をそろえてから」くらべる"""
    ma, mb = quick_metrics(a), quick_metrics(b)
    ca, va = ltas_third_octave(a)
    cb, vb = ltas_third_octave(b)
    n = min(len(ca), len(cb))
    ca, va, vb = ca[:n], va[:n], vb[:n]
    # 音の大きさをそろえる：1kHz付近（250Hz〜4kHz）の平均で合わせる（LUFSの差でも近いが、帯域の形をくらべやすい）
    mid = (ca >= 250) & (ca <= 4000)
    offset = float(np.mean(va[mid]) - np.mean(vb[mid]))
    vb_m = vb + offset
    diff = vb_m - va   # 正＝Bのほうが多い
    regions = []
    for name, lo, hi in REGIONS:
        la, lb = region_level(ca, va, lo, hi), region_level(ca, vb_m, lo, hi)
        if la is None or lb is None:
            continue
        d = round(lb - la, 1)
        regions.append({'name': name, 'range_hz': f'{lo}-{hi}', 'b_minus_a_db': d})
    numbers = {}
    for k in ma:
        va_, vb_ = ma[k], mb[k]
        entry = {'a': va_, 'b': vb_}
        if isinstance(va_, (int, float)) and isinstance(vb_, (int, float)) and va_ is not None and vb_ is not None:
            entry['b_minus_a'] = round(vb_ - va_, 3)
        numbers[k] = entry
    out = {
        'a': a.path, 'b': b.path,
        'numbers': numbers,
        'tonal_balance_regions': regions,
        'tonal_curve': {'centers_hz': ca.tolist(), 'a_db': np.round(va, 1).tolist(), 'b_db_level_matched': np.round(vb_m, 1).tolist(),
                        'b_minus_a_db': np.round(diff, 1).tolist()},
        'note': '音の形は 250Hz〜4kHz の平均で音量をそろえてからくらべた（大きいほうが良く聴こえる錯覚をのぞくため）。正＝Bのほうが多い。±2dB以上が目安で耳でわかる差',
    }
    if image_path:
        out['image'] = render_compare_image(a, b, ca, va, vb_m, diff, image_path)
    return out


def mix_doctor(a):
    """ミックスのお医者さん：経験則で気になるところを挙げる（好みや狙いでわざとやっていることもある。決めるのは耳）"""
    from scipy.signal import butter, sosfiltfilt
    import pyloudnorm as pyln
    findings = []

    def add(level, area, what, evidence, hint):
        findings.append({'level': level, 'area': area, 'what': what, 'evidence': evidence, 'hint': hint})

    # 1. 音の形：その曲自身の傾き（100Hz〜4kHzの回帰直線）から飛び出しているところ。基準はジャンルではなく「その曲の全体の傾き」
    from scipy.signal import welch
    f, P = welch(a.mono_native, fs=a.sr_native, nperseg=8192)
    centers, energy = [], []
    for fc in THIRD_OCT_CENTERS:
        lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
        if hi > a.sr_native / 2:
            break
        m = (f >= lo) & (f < hi)
        if m.any():
            centers.append(fc)
            energy.append(float(10 * np.log10(P[m].sum() + 1e-20)))   # 帯域のエネルギー（ピンクノイズで平ら）
    centers, energy = np.array(centers), np.array(energy)
    fit = (centers >= 100) & (centers <= 4000)   # 高域が急に落ちる曲（mp3の上の切れ・暗い音色）に傾きが引っぱられないよう 4kHz まで
    slope, icpt = np.polyfit(np.log2(centers[fit]), energy[fit], 1)
    resid = energy - (slope * np.log2(centers) + icpt)

    def res(lo, hi):
        m = (centers >= lo) & (centers < hi)
        return float(resid[m].mean()) if m.any() else 0.0

    r = {name: res(lo, hi) for name, lo, hi in REGIONS}
    r['空気感'] = res(10000, 15000)   # mp3 は16〜18kHzより上を切ることが多いので、空気感は15kHzまでで見る
    tilt_note = f'全体の傾き {slope:+.1f}dB/オクターブ（帯域エネルギー・ピンクノイズで0）'
    if r['こもり'] >= 3:
        add('気になる', 'こもり（200〜500Hz）', 'こもって聴こえやすい', f"曲の傾きより {r['こもり']:+.1f}dB 多い（{tilt_note}）", 'パッドやギター、ピアノの左手など和音楽器の200〜500Hzを少し削る')
    if r['鼻づまり感'] >= 3:
        add('気になる', '鼻づまり感（700〜1500Hz）', '箱っぽい・鼻にかかった感じになりやすい', f"曲の傾きより {r['鼻づまり感']:+.1f}dB 多い", '中域が集まっている楽器を探して少しだけ削る')
    if r['耳に痛いところ'] >= 3:
        add('要注意', '耳に痛いところ（2〜5kHz）', '長く聴くと疲れやすい', f"曲の傾きより {r['耳に痛いところ']:+.1f}dB 多い", 'リードやスネア、歪みの2〜5kHzを少し下げる。ゲームのBGMは長く流れるので特に効く')
    if r['歯擦音・シンバルの刺さり'] >= 3:
        add('気になる', '刺さり（5〜9kHz）', 'シンバルや歯擦音が刺さりやすい', f"曲の傾きより {r['歯擦音・シンバルの刺さり']:+.1f}dB 多い", 'ハイハット・シンバルの音量か、ディエッサー')
    if r['空気感'] <= -6:
        add('参考', '空気感（10〜15kHz）', '上のほうが少なめで、こもって・暗く聴こえることがある', f"曲の傾きより {r['空気感']:+.1f}dB（16kHzより上はmp3で切れることが多いので見ていない）", '狙いでなければ、高域を少し持ち上げる（シェルフ）')
    if r['超低域（体で感じる）'] >= 4:
        add('気になる', '超低域（20〜60Hz）', '低音がふくらんで、ほかの音を押しやすい', f"曲の傾きより {r['超低域（体で感じる）']:+.1f}dB 多い", '30Hz以下を切る、キックとサブベースの役割を分ける')
    if r['低音の芯'] <= -4:
        add('参考', '低音の芯（60〜150Hz）', '低音の芯が細く、軽く聴こえることがある', f"曲の傾きより {r['低音の芯']:+.1f}dB", 'ベースの80〜120Hzを足す')

    # 2. ピークと音割れ
    pk = peak_analysis(a)
    lufs = float(pyln.Meter(a.sr_native).integrated_loudness(a.stereo.T))
    if pk['clipped_samples'] > 0:
        add('要注意', '音割れ', '±0.999以上に張り付いたサンプルがある', f"{pk['clipped_samples']}サンプル", '全体を1〜2dB下げるか、リミッターの天井を下げる')
    if pk['true_peak_dbtp'] > -1.0:
        add('気になる' if pk['true_peak_dbtp'] <= 0 else '要注意', 'true peak', 'mp3やoggにしたり配信したりすると割れることがある',
            f"{pk['true_peak_dbtp']}dBTP（目安は -1dBTP 以下）", 'リミッターの天井を -1dBTP にする（true peak対応のリミッターで）')

    # 3. ダイナミクス（潰しすぎ）
    rms = float(np.sqrt(np.mean(a.mono_native ** 2)))
    crest = float(db(np.abs(a.mono_native).max()) - db(rms))
    plr = pk['true_peak_dbtp'] - lufs
    if crest < 8 or plr < 7:
        add('気になる', 'ダイナミクス', '音が潰れて、迫力より疲れが勝ちやすい', f"ピーク−平均 {crest:.1f}dB・true peak−LUFS {plr:.1f}dB（どちらも8前後より小さいと強く潰れている目安）", 'リミッターやコンプのかけすぎを戻す')

    # 4. 低音の左右の広がり・モノにしたときの減り
    if not a.is_mono_source:
        L, R = a.stereo[0], a.stereo[1]
        sos = butter(4, 150, 'lowpass', fs=a.sr_native, output='sos')
        Ll, Rl = sosfiltfilt(sos, L), sosfiltfilt(sos, R)
        mid_l, side_l = (Ll + Rl) / 2, (Ll - Rl) / 2
        side_ratio = float(np.sqrt(np.mean(side_l ** 2)) / (np.sqrt(np.mean(mid_l ** 2)) + 1e-12))
        corr_low = float(np.corrcoef(Ll, Rl)[0, 1])
        if side_ratio > 0.3 or corr_low < 0.5:
            add('要注意' if corr_low < 0.2 else '気になる', '低音の広がり（〜150Hz）', '低音が左右に広がっていて、スマホやモノラルで低音が減る・ぼやける',
                f"150Hz以下の 左右の差/真ん中 = {side_ratio:.2f}・相関 {corr_low:.2f}（低音は相関0.8以上・差0.3未満が目安）", '低音の楽器をモノにする（150Hz以下をモノにする道具を使う）')
        mono_drop = float(db(np.sqrt(np.mean(((L + R) / 2) ** 2))) - db((np.sqrt(np.mean(L ** 2)) + np.sqrt(np.mean(R ** 2))) / 2))
        if mono_drop < -3:
            add('気になる', 'モノラルにしたとき', '左右を混ぜると音が小さく・薄くなる（位相の打ち消し）', f"モノにすると {mono_drop:.1f}dB", '左右で逆相になっている楽器や、広げすぎのエフェクトを探す')
        bal = float(db(np.sqrt(np.mean(R ** 2))) - db(np.sqrt(np.mean(L ** 2))))
        if abs(bal) > 1.0:
            add('参考', '左右のバランス', '左右どちらかに寄っている', f"{'右' if bal > 0 else '左'}が {abs(bal):.1f}dB 大きい", 'パンや片側だけのエフェクトを見直す')

    # 5. DCずれ
    dc = max(abs(float(ch.mean())) for ch in a.stereo)
    if dc > 0.01:
        add('参考', 'DCずれ', '波形の中心が0からずれている', f"{dc:.3f}", 'ハイパス（20Hz前後）を入れる')

    order = {'要注意': 0, '気になる': 1, '参考': 2}
    findings.sort(key=lambda x: order[x['level']])
    return {
        'findings': findings,
        'tilt_db_per_octave': round(float(slope), 2),
        'region_residuals_db': {k: round(v, 1) for k, v in r.items()},
        'integrated_lufs': round(lufs, 2),
        'note': '経験則の目安。基準はジャンルの平均ではなく「その曲自身の全体の傾き」。狙いや好みでわざとそうしていることもあるので、決めるのは耳。楽器どうしのぶつかり（キックとベースなど）は見ていない',
    }


def _chord_templates():
    names, temps = [], []
    for root in range(12):
        for qual, ivs in (('', (0, 4, 7)), ('m', (0, 3, 7))):
            t = np.zeros(12)
            for k, iv in enumerate(ivs):
                t[(root + iv) % 12] = 1.0 if k != 2 else 0.8   # 5度は少し軽く（どのコードにもよく出るため）
            names.append(NOTES[root] + qual)
            temps.append(t / np.linalg.norm(t))
    return names, np.array(temps)


def chord_analysis(a, beats_per_bar=4):
    """コード進行の推定：拍ごとのクロマ（和音成分）を長三和音・短三和音の型とくらべ、ビタビでなめらかにして小節にまとめる"""
    y_h, _ = a.hpss()
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=a.sr, hop_length=hop, bins_per_octave=36)
    tempo, beat_times = a.beats()
    bf = librosa.time_to_frames(beat_times, sr=a.sr, hop_length=hop)
    bf = bf[bf < chroma.shape[1]]
    if len(bf) < 4:
        return {'note': '拍が少なすぎてコードを推定できない'}
    beat_times = beat_times[:len(bf)]
    sync = librosa.util.sync(chroma, np.concatenate([bf, [chroma.shape[1]]]), aggregate=np.median)[:, :len(bf)]
    energy = sync.sum(axis=0)
    names, T = _chord_templates()
    C = sync / (np.linalg.norm(sync, axis=0, keepdims=True) + 1e-9)
    sim = T @ C                                   # (24, 拍)
    # ビタビ：同じコードにとどまりやすくする
    n_states, n_beats = len(names), sim.shape[1]
    emit = sim * 12.0
    stay, move = np.log(0.80), np.log(0.20 / (n_states - 1))
    trans = np.full((n_states, n_states), move)
    np.fill_diagonal(trans, stay)
    score = emit[:, 0].copy()
    back = np.zeros((n_states, n_beats), dtype=int)
    for t in range(1, n_beats):
        cand = score[:, None] + trans
        back[:, t] = np.argmax(cand, axis=0)
        score = cand[back[:, t], np.arange(n_states)] + emit[:, t]
    path = np.zeros(n_beats, dtype=int)
    path[-1] = int(np.argmax(score))
    for t in range(n_beats - 1, 0, -1):
        path[t - 1] = back[path[t], t]
    quiet = energy < 0.15 * np.median(energy)
    beat_chords = ['N' if quiet[i] else names[path[i]] for i in range(n_beats)]

    # 小節の頭：コードが変わる拍がいちばん多くそろう位置を小節の頭にする
    changes = [i for i in range(1, n_beats) if beat_chords[i] != beat_chords[i - 1]]
    phase = max(range(beats_per_bar), key=lambda p: sum(1 for c in changes if (c - p) % beats_per_bar == 0))
    starts = list(range(phase, n_beats, beats_per_bar))
    if phase > 0:
        starts = [0] + starts
    bars = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else n_beats
        uniq = []
        for c in beat_chords[s:e]:
            if not uniq or uniq[-1] != c:
                uniq.append(c)
        bars.append({'start_sec': round(float(beat_times[s]), 2), 'chord': '/'.join(uniq)})
    bar_labels = [b['chord'] for b in bars]

    timeline = []
    for i, c in enumerate(beat_chords):
        if not timeline or timeline[-1]['chord'] != c:
            timeline.append({'start_sec': round(float(beat_times[i]), 2), 'chord': c})
    for i in range(len(timeline)):
        timeline[i]['end_sec'] = timeline[i + 1]['start_sec'] if i + 1 < len(timeline) else round(a.duration, 2)

    from collections import Counter
    def grams_of(n):
        # 同じコードがずっと続くだけの並びは「進行」ではないので数えない
        return Counter(tuple(bar_labels[i:i + n]) for i in range(len(bar_labels) - n + 1)
                       if 'N' not in bar_labels[i:i + n] and len(set(bar_labels[i:i + n])) >= 2)
    common = [{'bars': list(g), 'count': c} for g, c in grams_of(4).most_common(3) if c >= 2]
    common8 = [{'bars': list(g), 'count': c} for g, c in grams_of(8).most_common(2) if c >= 2]
    used = Counter(c for c in beat_chords if c != 'N')
    conf = float(np.mean([sim[path[i], i] for i in range(n_beats) if not quiet[i]])) if (~quiet).any() else 0.0
    return {
        'beats_per_bar_assumed': beats_per_bar,
        'bars': bars,
        'timeline': timeline,
        'most_common_4bars': common,
        'most_common_8bars': common8,
        'chords_used': [{'chord': c, 'beats': n} for c, n in used.most_common(8)],
        'match_mean': round(conf, 3),
        'note': '推定（長三和音・短三和音のみ。7th・テンション・分数コードは近い三和音になる）。拍子は4拍子と仮定。音だけからの推定なので、だいたい6〜7割の当たり方と思って使う。N＝和音がはっきりしない（静か・ドラムだけ）',
    }


def _setup_jp_font(plt, matplotlib):
    for fam in ('Yu Gothic', 'Meiryo', 'Noto Sans JP', 'Hiragino Sans', 'MS Gothic'):
        if any(f.name == fam for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams['font.family'] = fam
            return


def render_compare_image(a, b, centers, va, vb_m, diff, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _setup_jp_font(plt, matplotlib)
    name = lambda p: p.replace(chr(92), '/').split('/')[-1]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [2, 1.2]})
    fig.suptitle(f"A：{name(a.path)}　／　B：{name(b.path)}（音量をそろえて比較）", fontsize=13)
    for (nm, lo, hi), col in zip(REGIONS, ['#edf2f7', '#e6fffa', '#fefcbf', '#fff5f5', '#fed7d7', '#e9d8fd', '#ebf8ff']):
        for ax in (ax1, ax2):
            ax.axvspan(lo, hi, color=col, alpha=0.6, lw=0)
        ax1.text(np.sqrt(lo * hi), 1.0, nm, transform=ax1.get_xaxis_transform(), ha='center', va='bottom', fontsize=8, color='#4a5568')
    ax1.semilogx(centers, va, '-o', color='#2b6cb0', ms=3, lw=1.6, label='A')
    ax1.semilogx(centers, vb_m, '-o', color='#dd6b20', ms=3, lw=1.6, label='B（音量そろえ）')
    ax1.set_ylabel('平均の強さ dB')
    ax1.legend(loc='lower left')
    ax1.grid(alpha=0.3, which='both')
    ax2.semilogx(centers, diff, '-o', color='#2d3748', ms=3, lw=1.4)
    ax2.axhline(0, color='#718096', lw=1)
    for lim in (2, -2):
        ax2.axhline(lim, color='#e53e3e', lw=0.8, ls='--')
    ax2.set_ylabel('B − A（dB）\n正＝Bが多い')
    ax2.set_xlabel('周波数 Hz')
    ax2.grid(alpha=0.3, which='both')
    ticks = [30, 60, 100, 250, 500, 1000, 2000, 4000, 8000, 16000]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(['30', '60', '100', '250', '500', '1k', '2k', '4k', '8k', '16k'])
    ax2.set_xlim(20, 20000)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def render_image(a, out_path, structure=None, t_from=None, t_to=None):
    """音を絵にする（AIが画像として「見る」ため）：音量・スペクトログラム・和音/打楽器・区間を時間で並べる"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import librosa.display
    for fam in ('Yu Gothic', 'Meiryo', 'Noto Sans JP', 'Hiragino Sans', 'MS Gothic'):
        if any(f.name == fam for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams['font.family'] = fam
            break

    t0 = 0.0 if t_from is None else max(0.0, t_from)
    t1 = a.duration if t_to is None else min(a.duration, t_to)
    if t1 - t0 < 0.5:
        raise ValueError('絵にする範囲が短すぎる')
    srn = a.sr_native
    x = a.mono_native[int(t0 * srn):int(t1 * srn)]
    st = a.stereo[:, int(t0 * srn):int(t1 * srn)]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={'height_ratios': [1.1, 3.2, 1.1]})
    tempo, beat_times = a.beats()
    title = (f"{a.path.replace(chr(92), '/').split('/')[-1]}　{t0:.1f}〜{t1:.1f}秒"
             f"　テンポ約{tempo:.0f}BPM")
    fig.suptitle(title, fontsize=14)

    # 1. 音量（約50msごとのRMS）と音割れ
    hop = max(1, int(srn * 0.05))
    n = len(x) // hop
    rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1))
    tt = t0 + (np.arange(n) + 0.5) * hop / srn
    ax = axes[0]
    ax.plot(tt, db(rms), color='#2b6cb0', lw=1)
    ax.set_ylabel('音量 dB')
    ax.set_ylim(max(-60, float(db(rms).min()) - 3), 3)
    clip_idx = np.nonzero(np.abs(st).max(axis=0) >= 0.999)[0]
    if len(clip_idx):
        ct = t0 + clip_idx / srn
        ax.plot(ct, np.full_like(ct, 1.5), '|', color='#e53e3e', ms=10)
        ax.text(0.995, 0.92, f'赤＝音割れ {len(clip_idx)}サンプル', transform=ax.transAxes,
                ha='right', va='top', color='#e53e3e', fontsize=10)
    ax.grid(alpha=0.3)

    # 2. メルスペクトログラム（元のサンプルレート）
    ax = axes[1]
    S = librosa.feature.melspectrogram(y=x, sr=srn, n_fft=4096, hop_length=1024, n_mels=160, fmin=20, fmax=srn / 2)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=srn, hop_length=1024, x_axis='time', y_axis='mel', fmin=20, fmax=srn / 2,
                             ax=ax, cmap='magma', x_coords=t0 + librosa.frames_to_time(np.arange(S_db.shape[1]), sr=srn, hop_length=1024))
    ax.set_yticks([60, 250, 500, 1000, 2000, 4000, 8000, 16000])
    ax.set_yticklabels(['60', '250', '500', '1k', '2k', '4k', '8k', '16k'])
    ax.set_ylabel('高さ Hz（明るいほど強い）')
    ax.set_xlabel('')

    # 3. 和音と打楽器（HPSS・22kHz）
    ax = axes[2]
    y_h, y_p = a.hpss()
    s0, s1 = int(t0 * a.sr), int(t1 * a.sr)
    hop2 = 1024
    for sig, name, col in ((y_h[s0:s1], '和音', '#38a169'), (y_p[s0:s1], '打楽器', '#dd6b20')):
        r = librosa.feature.rms(y=sig, hop_length=hop2).flatten()
        tr = t0 + librosa.frames_to_time(np.arange(len(r)), sr=a.sr, hop_length=hop2)
        ax.plot(tr, db(r), color=col, lw=1, label=name)
    ax.set_ylabel('dB')
    ax.set_ylim(-70, 0)
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xlabel('時間（分:秒）')

    # 区間の切れ目と、狭い範囲なら拍も
    if structure:
        for sec in structure['sections']:
            s = sec['start_sec']
            if t0 <= s <= t1:
                for axx in axes:
                    axx.axvline(s, color='#00e5ff', lw=1.2, alpha=0.9)
                mid = (max(s, t0) + min(sec['end_sec'], t1)) / 2
                visible = min(sec['end_sec'], t1) - max(s, t0)
                if t0 <= mid <= t1 and visible >= 0.07 * (t1 - t0):   # 狭い区間は名前を出さない（重なって読めないため）
                    axes[0].text(mid, 0.02, sec['label'], transform=axes[0].get_xaxis_transform(),
                                 ha='center', va='bottom', fontsize=9, color='#1a202c',
                                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
    if t1 - t0 <= 30:
        for b in beat_times:
            if t0 <= b <= t1:
                axes[2].axvline(b, color='#a0aec0', lw=0.6, alpha=0.6)
    axes[2].set_xlim(t0, t1)
    # 横軸は「分:秒」で（specshow の時間表示は範囲をずらすと 1e+02 のように崩れるため、自分で書く）
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    span = t1 - t0
    fmt = (lambda v, _: f"{int(v // 60)}:{v % 60:04.1f}") if span <= 60 else (lambda v, _: f"{int(v // 60)}:{int(round(v % 60)) % 60:02d}")
    for axx in axes:
        axx.xaxis.set_major_locator(MaxNLocator(nbins=12))
        axx.xaxis.set_major_formatter(FuncFormatter(fmt))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def summary_ja(result):
    """ニコイが読んで、そのまま話せる日本語の短いまとめ"""
    lines = []
    b = result['basic']
    lines.append(f"長さ {result['analyzed_duration_sec']}秒（{result['scope']}）。テンポ約{b['tempo_bpm']:.0f}BPM、キー {b['key']['note']} {'マイナー' if b['key']['mode'] == 'minor' else 'メジャー'}（確からしさ {b['key']['confidence']}）")
    lo = b['loudness']
    clip_note = 'なし' if lo['clipped_samples'] == 0 else f"{lo['clipped_samples']}サンプル"
    tp_warn = '（0を超えている：mp3にすると割れるかも）' if lo['true_peak_dbtp'] > 0 else ''
    lines.append(f"ピーク {lo['peak_db']}dBFS・true peak {lo['true_peak_dbtp']}dBTP{tp_warn}・音割れ {clip_note}")
    if 'lufs' in result:
        lines.append(f"音の大きさ {result['lufs']['integrated_lufs']} LUFS（Spotify基準-14との差 {result['lufs']['headroom_to_targets_db']['Spotify']:+}）")
    if 'bands' in result:
        bd = result['bands']
        low = bd['sub_bass_20_60hz']['percent'] + bd['bass_60_250hz']['percent']
        lines.append(f"帯域：低音(〜250Hz) {low:.1f}%・中域(500〜2k) {bd['mid_500_2000hz']['percent']}%・高域(6k〜) {bd['brilliance_6000_20000hz']['percent']}%（うち10k〜 {bd['air_10000_20000hz_within_brilliance']['percent']}%）")
    if 'hpss' in result:
        lines.append(f"和音 {result['hpss']['harmonic_ratio'] * 100:.0f}%・打楽器 {result['hpss']['percussive_ratio'] * 100:.0f}%")
    if 'structure' in result:
        st = result['structure']
        secs = '／'.join(f"{s['start_sec']}〜{s['end_sec']}秒 {s['label']}（打{s['percussive_ratio'] * 100:.0f}%）" for s in st['sections'])
        lines.append(f"区間：{secs}")
        if st['percussion_enters_sec'] is not None:
            lines.append(f"打楽器がはっきり入ってくるのは {st['percussion_enters_sec']}秒ごろ")
        else:
            lines.append("打楽器はほとんど入っていない")
    if 'tags' in result:
        tg = result['tags']
        inst = '・'.join(t['tag'] for t in tg['instruments'][:5])
        mood = '・'.join(t['tag'] for t in tg['moods'][:3])
        lines.append(f"聴こえる楽器（機械の耳）：{inst or 'はっきりせず'}。ムード：{mood or 'はっきりせず'}")
    if 'stereo' in result and 'L_R_correlation' in result['stereo']:
        s = result['stereo']
        lines.append(f"ステレオ：左右の相関 {s['L_R_correlation']}・広がり {s['stereo_width_ratio']}・左右差 {s['lr_balance_db']}dB")
    if 'game' in result:
        g = result['game']
        lines.append(f"頭の無音 {g['leading_silence_sec']}秒・お尻の無音 {g['trailing_silence_sec']}秒・DCずれ {g['dc_offset_LR']}")
        if g['loop_candidates']:
            c = g['loop_candidates'][0]
            lines.append(f"ループ候補のいちばん：{c['start_sec']}→{c['end_sec']}秒（{c['length_sec']}秒・似かた {c['similarity']}）")
    if 'chords' in result and 'bars' in result['chords']:
        ch = result['chords']
        first = ' | '.join(b['chord'] for b in ch['bars'][:8])
        lines.append(f"コード（推定・4拍子と仮定）：最初の8小節 {first}")
        if ch.get('most_common_8bars'):
            mc = ch['most_common_8bars'][0]
            lines.append(f"いちばん繰り返される8小節：{' | '.join(mc['bars'])}（{mc['count']}回）")
        if ch['most_common_4bars']:
            mc = ch['most_common_4bars'][0]
            lines.append(f"いちばん繰り返される4小節：{' | '.join(mc['bars'])}（{mc['count']}回）")
        lines.append('よく出るコード：' + '・'.join(c['chord'] for c in ch['chords_used'][:6]))
    if 'doctor' in result:
        d = result['doctor']
        lines.append(f"―― ミックスのお医者さん（全体の傾き {d['tilt_db_per_octave']:+}dB/oct）――")
        if not d['findings']:
            lines.append('目立つ問題は見つからなかった（見ている項目：帯域の飛び出し・音割れ・true peak・潰しすぎ・低音の広がり・モノ・左右・DC）')
        for f_ in d['findings']:
            lines.append(f"[{f_['level']}] {f_['area']}：{f_['what']}（{f_['evidence']}）→ {f_['hint']}")
    if 'compare' in result:
        c = result['compare']
        nm = c['numbers']
        lines.append('―― AとBの比較（B−A）――')
        lines.append(f"音の大きさ {nm['integrated_lufs']['a']} → {nm['integrated_lufs']['b']} LUFS（{nm['integrated_lufs']['b_minus_a']:+}）・true peak {nm['true_peak_dbtp']['a']} → {nm['true_peak_dbtp']['b']}・音割れ {nm['clipped_samples']['a']} → {nm['clipped_samples']['b']}")
        lines.append(f"テンポ {nm['tempo_bpm']['a']} → {nm['tempo_bpm']['b']}・キー {nm['key']['a']} → {nm['key']['b']}・打楽器 {nm['percussive_ratio']['a']} → {nm['percussive_ratio']['b']}・明るさ {nm['brightness_centroid_hz']['a']} → {nm['brightness_centroid_hz']['b']}Hz")
        lines.append(f"ダイナミクス（ピーク−平均） {nm['crest_factor_db']['a']} → {nm['crest_factor_db']['b']}dB・ステレオの広がり {nm['stereo_width']['a']} → {nm['stereo_width']['b']}")
        big = [r for r in c['tonal_balance_regions'] if abs(r['b_minus_a_db']) >= 2]
        if big:
            lines.append('音の形（音量をそろえて）：' + '・'.join(f"{r['name']}（{r['range_hz']}Hz）が{'Bのほうが' if r['b_minus_a_db'] > 0 else 'Aのほうが'}{abs(r['b_minus_a_db'])}dB多い" for r in big))
        else:
            lines.append('音の形（音量をそろえて）：どの帯域も差は2dB未満。だいたい同じバランス')
        if 'image' in c:
            lines.append(f"比較の絵：{c['image']}")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('audio_path')
    p.add_argument('--offset', type=float, default=0)
    p.add_argument('--duration', type=float, default=None, help='聴く長さ（秒）。省略すると全長')
    p.add_argument('--full', action='store_true', help='全長分析（いまは既定で全長。互換のため残す）')
    p.add_argument('--deep', action='store_true', help='音色・和声追加')
    p.add_argument('--structure', action='store_true', help='構造検出（区間ごとの中身つき）')
    p.add_argument('--rhythm', action='store_true', help='リズム詳細')
    p.add_argument('--bands', action='store_true', help='周波数帯別')
    p.add_argument('--hpss', action='store_true', help='和声/打音分離')
    p.add_argument('--pitch', action='store_true', help='F0追跡')
    p.add_argument('--tags', action='store_true', help='PANNsジャンル/楽器タグ（全長を10秒ごと）')
    p.add_argument('--game', action='store_true', help='ゲームBGM用チェック（無音・DC・ループ候補）')
    p.add_argument('--whisper', action='store_true', help='歌詞文字起こし(base)')
    p.add_argument('--whisper-large', action='store_true', dest='whisper_large', help='歌詞文字起こし(large・高精度)')
    p.add_argument('--lang', default=None, help='Whisperの言語（ja/en など）。省略で自動判定')
    p.add_argument('--lufs', action='store_true', help='LUFSラウドネス測定')
    p.add_argument('--stereo', action='store_true', help='ステレオ分析')
    p.add_argument('--demucs', action='store_true', help='楽器分離(vocals/drums/bass/other)')
    p.add_argument('--all', action='store_true', help='全部（Whisper large と Demucs 以外）')
    p.add_argument('--summary', action='store_true', help='日本語の短いまとめだけ出す（--structure --bands --hpss --tags --lufs --stereo --game をまとめて有効にする）')
    p.add_argument('--image', nargs='?', const='auto', default=None,
                   help='音を絵（PNG）にする。パス省略で「<曲名>_ears.png」を曲と同じフォルダに')
    p.add_argument('--image-from', type=float, default=None, help='絵にする範囲の始まり（秒）')
    p.add_argument('--image-to', type=float, default=None, help='絵にする範囲の終わり（秒）')
    p.add_argument('--chords', action='store_true', help='コード進行の推定（小節ごと・いちばん繰り返される4小節）')
    p.add_argument('--doctor', action='store_true', help='ミックスのお医者さん（こもり・耳に痛い・低音の広がり・潰しすぎ・true peak など）')
    p.add_argument('--compare', default=None, help='くらべる曲B（Aはaudio_path）。音量をそろえて音の形・数字の差を出し、<A>_vs_<B>.png を描く')
    args = p.parse_args()

    if args.summary:
        args.structure = args.bands = args.hpss = args.tags = args.lufs = args.stereo = args.game = args.doctor = True

    duration = None if args.full else args.duration
    a = Audio(args.audio_path, args.offset, duration)

    result = {
        'file': args.audio_path,
        'analyzed_offset_sec': args.offset,
        'analyzed_duration_sec': round(a.duration, 2),
        'scope': '全長' if duration is None and args.offset == 0 else f'{args.offset}秒から{round(a.duration, 1)}秒ぶん',
        'sample_rate_native': a.sr_native,
        'sample_rate_analysis': a.sr,
        'basic': basic_analysis(a),
    }
    if args.deep or args.all:
        result['deep'] = deep_analysis(a)
    if args.structure or args.all:
        result['structure'] = structure_analysis(a)
    if args.rhythm or args.all:
        result['rhythm'] = rhythm_analysis(a)
    if args.bands or args.all:
        result['bands'] = bands_analysis(a)
    if args.hpss or args.all:
        result['hpss'] = hpss_analysis(a)
    if args.pitch or args.all:
        result['pitch'] = pitch_analysis(a)
    if args.tags or args.all:
        result['tags'] = panns_tags(a)
    if args.lufs or args.all:
        result['lufs'] = lufs_analysis(a)
    if args.stereo or args.all:
        result['stereo'] = stereo_analysis(a)
    if args.game or args.all:
        result['game'] = game_checks(a)
    if args.demucs:
        result['demucs'] = demucs_separate(args.audio_path)
    if args.whisper or args.all:
        result['whisper'] = whisper_lyrics(args.audio_path, args.offset, duration, model_size='base', lang=args.lang)
    if args.whisper_large:
        result['whisper'] = whisper_lyrics(args.audio_path, args.offset, duration, model_size='large', lang=args.lang)

    if args.doctor or args.all:
        result['doctor'] = mix_doctor(a)
    if args.chords or args.all:
        result['chords'] = chord_analysis(a)

    if args.compare:
        import os
        b = Audio(args.compare, 0, None)
        sa = os.path.splitext(os.path.basename(args.audio_path))[0]
        sb = os.path.splitext(os.path.basename(args.compare))[0]
        cmp_img = os.path.join(os.path.dirname(args.audio_path) or '.', f'{sa}_vs_{sb}.png')
        result['compare'] = compare_analysis(a, b, cmp_img)

    if args.image:
        import os
        if args.image == 'auto':
            base = os.path.splitext(args.audio_path)[0]
            rng = '' if args.image_from is None and args.image_to is None else f"_{args.image_from or 0:g}-{args.image_to if args.image_to is not None else round(a.duration):g}s"
            img_path = f'{base}_ears{rng}.png'
        else:
            img_path = args.image
        st = result.get('structure') or structure_analysis(a)
        result['image'] = render_image(a, img_path, st, args.image_from, args.image_to)

    if args.summary:
        text = summary_ja(result)
        if 'image' in result:
            text += f"\n絵：{result['image']}"
        print(text)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder))


def whisper_lyrics(path, offset=0, duration=None, model_size='base', lang=None):
    """歌詞文字起こし（Whisper、librosa経由でffmpeg不要）。言語は自動判定（--lang で指定）"""
    import whisper
    y, sr = librosa.load(path, sr=16000, mono=True, offset=offset, duration=duration)
    with contextlib.redirect_stdout(sys.stderr):
        model = whisper.load_model(model_size)
        result = model.transcribe(y.astype(np.float32), language=lang, verbose=False)
    return {
        'language': result.get('language'),
        'full_text': result.get('text', '').strip(),
        'segments': [{'start': round(s['start'], 2), 'end': round(s['end'], 2), 'text': s['text'].strip()}
                     for s in result.get('segments', [])],
        'model': f'whisper-{model_size}',
    }


if __name__ == '__main__':
    main()
