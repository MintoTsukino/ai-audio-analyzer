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
    args = p.parse_args()

    if args.summary:
        args.structure = args.bands = args.hpss = args.tags = args.lufs = args.stereo = args.game = True

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

    if args.summary:
        print(summary_ja(result))
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
