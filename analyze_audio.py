#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""音楽ファイル詳細分析スクリプト

使い方:
  python analyze_audio.py <音声ファイル>                    # 基本
  python analyze_audio.py <音声ファイル> --deep             # 音色・和声追加
  python analyze_audio.py <音声ファイル> --structure        # 構造検出
  python analyze_audio.py <音声ファイル> --rhythm           # リズム詳細
  python analyze_audio.py <音声ファイル> --bands            # 周波数帯別
  python analyze_audio.py <音声ファイル> --hpss             # 和声/打音分離
  python analyze_audio.py <音声ファイル> --pitch            # F0追跡
  python analyze_audio.py <音声ファイル> --whisper          # 歌詞文字起こし
  python analyze_audio.py <音声ファイル> --all              # 全部
  python analyze_audio.py <音声ファイル> --offset 60 --duration 30
  python analyze_audio.py <音声ファイル> --full             # 全長分析
"""
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import librosa.feature

sys.stdout.reconfigure(encoding='utf-8')


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


def basic_analysis(y, sr):
    """基本：テンポ・キー・スペクトル・音量"""
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key_note, key_mode, key_conf = estimate_key(chroma_mean)

    spec_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    spec_rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
    spec_bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())
    rms = float(librosa.feature.rms(y=y).mean())
    peak_db = 20 * np.log10(np.max(np.abs(y)) + 1e-10)
    rms_db = 20 * np.log10(rms + 1e-10)

    return {
        'tempo_bpm': float(tempo) if np.isscalar(tempo) else float(tempo[0]),
        'beats_count': int(len(beats)),
        'key': {'note': key_note, 'mode': key_mode, 'confidence': round(key_conf, 3)},
        'spectral': {
            'centroid_hz': round(spec_centroid, 1),
            'rolloff_hz': round(spec_rolloff, 1),
            'bandwidth_hz': round(spec_bandwidth, 1),
            'zero_crossing_rate': round(zcr, 4),
        },
        'loudness': {
            'peak_db': round(float(peak_db), 2),
            'rms_db': round(float(rms_db), 2),
            'dynamic_range_db': round(float(peak_db - rms_db), 2),
        },
    }


def deep_analysis(y, sr):
    """音色・和声詳細：MFCC・トンネッツ・スペクトラルコントラスト"""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1).round(2).tolist()
    mfcc_std = mfcc.std(axis=1).round(2).tolist()

    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_mean = tonnetz.mean(axis=1).round(3).tolist()

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = contrast.mean(axis=1).round(2).tolist()

    # クロマ全12音
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    chroma_normalized = (chroma / chroma.max()).round(3).tolist()

    return {
        'mfcc': {
            'mean': mfcc_mean,
            'std': mfcc_std,
            'note': 'MFCC 13次元の音色指紋。1次元目は音量、2-13は音色特性',
        },
        'tonnetz': {
            'mean': tonnetz_mean,
            'note': '和声的特徴。[5度,短3度,長3度] x [円x,円y] の6次元',
        },
        'spectral_contrast': {
            'mean': contrast_mean,
            'note': '7帯域それぞれの「ピーク-谷」のdB差。値が高いほどメリハリある',
        },
        'chroma_distribution': {
            'notes': NOTES,
            'normalized': chroma_normalized,
            'note': '各音の出現強度。最大値を1.0に正規化',
        },
    }


def structure_analysis(y, sr):
    """構造検出：セクション境界・エネルギー曲線"""
    # 構造境界（recurrence matrix 簡易版）
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=2048)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=2048)
    features = np.vstack([chroma, mfcc])

    # ノベルティ曲線
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=2048)
    bounds = librosa.segment.agglomerative(features, k=6)
    bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=2048).tolist()

    # エネルギー曲線（RMS）
    rms = librosa.feature.rms(y=y, hop_length=2048).flatten()
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=2048)
    energy_db = (20 * np.log10(rms + 1e-10)).round(2).tolist()

    return {
        'section_boundaries_sec': [round(t, 2) for t in bound_times],
        'sections_count': len(bound_times),
        'energy_curve_db': {
            'times_sec': [round(t, 2) for t in rms_times.tolist()],
            'values_db': energy_db,
            'note': '時間ごとのRMSラウドネス。サビは通常高め',
        },
        'onset_strength_max': round(float(onset_env.max()), 2),
        'onset_strength_mean': round(float(onset_env.mean()), 2),
    }


def rhythm_analysis(y, sr):
    """リズム詳細：オンセット・テンポグラム"""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).round(3).tolist()

    # テンポグラム
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    tempo_per_frame = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
    tempo_stability = float(np.std(tempo_per_frame))

    return {
        'onset_count': len(onset_times),
        'onset_density_per_sec': round(len(onset_times) / (len(y) / sr), 2),
        'first_10_onsets_sec': onset_times[:10],
        'tempo_stability_std': round(tempo_stability, 2),
        'tempo_stability_note': '低いほど安定。10以下なら安定、30以上は揺れ大',
    }


def bands_analysis(y, sr):
    """周波数帯別エネルギー：Bass/LowMid/Mid/HighMid/High"""
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    bands = {
        'sub_bass_20_60hz': (20, 60),
        'bass_60_250hz': (60, 250),
        'low_mid_250_500hz': (250, 500),
        'mid_500_2000hz': (500, 2000),
        'high_mid_2000_4000hz': (2000, 4000),
        'presence_4000_6000hz': (4000, 6000),
        'brilliance_6000_20000hz': (6000, 20000),
    }

    result = {}
    total = stft.sum()
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        energy = float(stft[mask].sum())
        ratio = energy / total if total > 0 else 0
        result[name] = {
            'energy_ratio': round(ratio, 4),
            'percent': round(ratio * 100, 1),
        }
    return result


def hpss_analysis(y, sr):
    """和声/打音分離"""
    y_harm, y_perc = librosa.effects.hpss(y)
    h_energy = float(np.sum(y_harm ** 2))
    p_energy = float(np.sum(y_perc ** 2))
    total = h_energy + p_energy
    return {
        'harmonic_ratio': round(h_energy / total, 3),
        'percussive_ratio': round(p_energy / total, 3),
        'note': '和声成分（メロディ・コード）と打音成分（ドラム）の比率',
    }


def pitch_analysis(y, sr):
    """F0追跡（pyin）"""
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
    )
    voiced_f0 = f0[~np.isnan(f0)]
    if len(voiced_f0) == 0:
        return {'note': 'F0検出失敗（モノフォニック音源向け）'}

    pitches_hz = voiced_f0
    pitches_midi = librosa.hz_to_midi(pitches_hz)
    pitches_note = [librosa.midi_to_note(int(round(m))) for m in pitches_midi[::10]][:30]

    return {
        'f0_count': int(len(voiced_f0)),
        'f0_mean_hz': round(float(np.mean(pitches_hz)), 2),
        'f0_median_hz': round(float(np.median(pitches_hz)), 2),
        'f0_min_hz': round(float(np.min(pitches_hz)), 2),
        'f0_max_hz': round(float(np.max(pitches_hz)), 2),
        'pitch_range_semitones': round(float(np.max(pitches_midi) - np.min(pitches_midi)), 1),
        'voicing_ratio': round(float(np.sum(~np.isnan(f0)) / len(f0)), 3),
        'first_30_notes': pitches_note,
        'note': 'pyinによるモノフォニック音高追跡。ポリフォニックには弱い',
    }


def lufs_analysis(path, offset=0, duration=None):
    """LUFS ラウドネス測定（プロマスタリング基準）"""
    import pyloudnorm as pyln
    # ステレオ読み込み（LUFSはステレオ計測）
    y, sr = librosa.load(path, sr=None, mono=False, offset=offset, duration=duration)
    if y.ndim == 1:
        y = np.stack([y, y])
    y_t = y.T  # (samples, channels)
    meter = pyln.Meter(sr)
    integrated = meter.integrated_loudness(y_t)
    # 短期ラウドネス（3秒窓）
    block_samples = int(sr * 3.0)
    stl = []
    for i in range(0, len(y_t) - block_samples, block_samples // 2):
        block = y_t[i:i + block_samples]
        try:
            stl.append(meter.integrated_loudness(block))
        except Exception:
            pass

    target_specs = {
        'Spotify': -14.0,
        'YouTube': -14.0,
        'Apple Music': -16.0,
        'Tidal': -14.0,
        'CD/Streaming Pro': -10.0,
    }
    headroom_to_targets = {k: round(integrated - v, 2) for k, v in target_specs.items()}

    return {
        'integrated_lufs': round(float(integrated), 2),
        'short_term_lufs_min': round(float(min(stl)), 2) if stl else None,
        'short_term_lufs_max': round(float(max(stl)), 2) if stl else None,
        'short_term_lufs_range': round(float(max(stl) - min(stl)), 2) if stl else None,
        'headroom_to_targets_db': headroom_to_targets,
        'note': '負の値は基準より静か、正は大きい。Spotify基準は-14 LUFS',
    }


def stereo_analysis(path, offset=0, duration=None):
    """ステレオイメージ分析（L/R・位相・幅）"""
    y, sr = librosa.load(path, sr=None, mono=False, offset=offset, duration=duration)
    if y.ndim == 1:
        return {'note': 'モノラル音源なのでステレオ分析不可'}

    L, R = y[0], y[1]
    mid = (L + R) / 2
    side = (L - R) / 2

    mid_rms = float(np.sqrt(np.mean(mid ** 2)))
    side_rms = float(np.sqrt(np.mean(side ** 2)))
    L_rms = float(np.sqrt(np.mean(L ** 2)))
    R_rms = float(np.sqrt(np.mean(R ** 2)))

    correlation = float(np.corrcoef(L, R)[0, 1])
    stereo_width = side_rms / (mid_rms + 1e-10)

    # L/Rバランス
    lr_balance_db = 20 * np.log10((R_rms + 1e-10) / (L_rms + 1e-10))

    return {
        'L_R_correlation': round(correlation, 3),
        'L_R_correlation_note': '1.0=モノ、0=完全ステレオ、-1=位相反転（要注意）',
        'stereo_width_ratio': round(float(stereo_width), 3),
        'stereo_width_note': '0=モノ、1.0前後=広い、大きすぎはリスニング疲れる',
        'lr_balance_db': round(float(lr_balance_db), 2),
        'lr_balance_note': '正=右寄り、負=左寄り、0付近=センター。±0.5dB以内が理想',
        'mid_energy_db': round(20 * np.log10(mid_rms + 1e-10), 2),
        'side_energy_db': round(20 * np.log10(side_rms + 1e-10), 2),
    }


def demucs_separate(path, offset=0, duration=None, out_dir=None):
    """Demucs で楽器分離（vocal/drums/bass/other）"""
    import subprocess
    import os
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(path) or '.', 'demucs_out')
    os.makedirs(out_dir, exist_ok=True)

    cmd = ['python', '-m', 'demucs', '-o', out_dir, path]
    if duration:
        # Demucs は --duration なしなので、事前に短いファイル作る方が良いが、
        # ここでは全体処理
        pass

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {'error': result.stderr[-500:], 'note': 'Demucs実行失敗'}

    # 出力ファイル一覧
    name = os.path.splitext(os.path.basename(path))[0]
    sep_dir = os.path.join(out_dir, 'htdemucs', name)
    stems = {}
    if os.path.isdir(sep_dir):
        for f in os.listdir(sep_dir):
            stems[os.path.splitext(f)[0]] = os.path.join(sep_dir, f)

    # 各stemのエネルギー測定
    stem_energy = {}
    for stem, path_stem in stems.items():
        y_s, sr_s = librosa.load(path_stem, sr=22050, mono=True, duration=30)
        rms = float(np.sqrt(np.mean(y_s ** 2)))
        stem_energy[stem] = {
            'file': path_stem,
            'rms_db': round(20 * np.log10(rms + 1e-10), 2),
        }

    return {
        'stems_dir': sep_dir,
        'stems': stem_energy,
        'note': 'vocals/drums/bass/other に分離。stem毎にRMSエネルギー測定',
    }


def panns_tags(path, offset=0, duration=30, top_n=40):
    """PANNs 音響タグ付け（ジャンル・楽器・音響イベント）"""
    from panns_inference import AudioTagging, labels as panns_labels
    y, sr = librosa.load(path, sr=32000, mono=True, offset=offset, duration=duration)
    y_batch = y[None, :]
    at = AudioTagging(checkpoint_path=None, device='cpu')
    clipwise_output, _ = at.inference(y_batch)
    scores = clipwise_output[0]
    top_idx = np.argsort(scores)[::-1][:top_n]
    tags = [{'tag': panns_labels[i], 'score': round(float(scores[i]), 4)} for i in top_idx]

    # カテゴリ別集計
    genre_keywords = ['music', 'rock', 'pop', 'jazz', 'classical', 'hip hop', 'electronic',
                      'metal', 'punk', 'folk', 'country', 'blues', 'funk', 'disco', 'reggae',
                      'soul', 'r&b', 'house', 'techno', 'trance', 'dubstep', 'dance', 'ambient',
                      'soundtrack', 'orchestra']
    instrument_keywords = ['guitar', 'drum', 'bass', 'piano', 'violin', 'cello', 'flute',
                           'saxophone', 'trumpet', 'synthesizer', 'organ', 'harp', 'banjo',
                           'percussion', 'cymbal', 'snare', 'hi-hat', 'sampler', 'string',
                           'wind', 'brass', 'keyboard']
    vocal_keywords = ['singing', 'speech', 'vocal', 'shout', 'scream', 'rap', 'choir', 'whisper']
    mood_keywords = ['happy', 'sad', 'aggressive', 'chill', 'exciting', 'scary', 'tense',
                     'funny', 'cacophony']

    def collect(keywords):
        out = []
        for t in tags:
            tag_lower = t['tag'].lower()
            if any(k in tag_lower for k in keywords):
                out.append(t)
        return out[:10]

    return {
        'top_tags': tags[:15],
        'genres': collect(genre_keywords),
        'instruments': collect(instrument_keywords),
        'vocals': collect(vocal_keywords),
        'moods': collect(mood_keywords),
        'note': 'PANNs Cnn14 (AudioSet 527 tags)。スコアは0-1、相対的に解釈',
    }


def whisper_lyrics(path, offset=0, duration=None, model_size='base'):
    """歌詞文字起こし（Whisper、librosa経由でffmpeg不要）"""
    import whisper
    # librosa で読んで Whisper の仕様（16kHz mono float32）に変換
    y, sr = librosa.load(path, sr=16000, mono=True, offset=offset, duration=duration)
    audio = y.astype(np.float32)

    model = whisper.load_model(model_size)
    result = model.transcribe(audio, language='ja', verbose=False)
    segments = []
    for seg in result.get('segments', []):
        segments.append({
            'start': round(seg['start'], 2),
            'end': round(seg['end'], 2),
            'text': seg['text'].strip(),
        })
    return {
        'language': result.get('language'),
        'full_text': result.get('text', '').strip(),
        'segments': segments,
        'model': f'whisper-{model_size}',
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('audio_path')
    p.add_argument('--offset', type=float, default=0)
    p.add_argument('--duration', type=float, default=30)
    p.add_argument('--full', action='store_true', help='全長分析')
    p.add_argument('--deep', action='store_true', help='音色・和声追加')
    p.add_argument('--structure', action='store_true', help='構造検出')
    p.add_argument('--rhythm', action='store_true', help='リズム詳細')
    p.add_argument('--bands', action='store_true', help='周波数帯別')
    p.add_argument('--hpss', action='store_true', help='和声/打音分離')
    p.add_argument('--pitch', action='store_true', help='F0追跡')
    p.add_argument('--tags', action='store_true', help='PANNsジャンル/楽器タグ')
    p.add_argument('--whisper', action='store_true', help='歌詞文字起こし(base)')
    p.add_argument('--whisper-large', action='store_true', dest='whisper_large', help='歌詞文字起こし(large・高精度)')
    p.add_argument('--lufs', action='store_true', help='LUFSラウドネス測定')
    p.add_argument('--stereo', action='store_true', help='ステレオ分析')
    p.add_argument('--demucs', action='store_true', help='楽器分離(vocals/drums/bass/other)')
    p.add_argument('--all', action='store_true', help='全部')
    args = p.parse_args()

    duration = None if args.full else args.duration
    y, sr = librosa.load(args.audio_path, sr=22050, mono=True, offset=args.offset, duration=duration)
    actual_duration = len(y) / sr

    result = {
        'file': args.audio_path,
        'analyzed_offset_sec': args.offset,
        'analyzed_duration_sec': round(actual_duration, 2),
        'sample_rate': sr,
        'basic': basic_analysis(y, sr),
    }

    if args.deep or args.all:
        result['deep'] = deep_analysis(y, sr)
    if args.structure or args.all:
        result['structure'] = structure_analysis(y, sr)
    if args.rhythm or args.all:
        result['rhythm'] = rhythm_analysis(y, sr)
    if args.bands or args.all:
        result['bands'] = bands_analysis(y, sr)
    if args.hpss or args.all:
        result['hpss'] = hpss_analysis(y, sr)
    if args.pitch or args.all:
        result['pitch'] = pitch_analysis(y, sr)
    if args.tags or args.all:
        result['tags'] = panns_tags(args.audio_path, args.offset, duration if duration else 30)
    if args.lufs or args.all:
        result['lufs'] = lufs_analysis(args.audio_path, args.offset, duration)
    if args.stereo or args.all:
        result['stereo'] = stereo_analysis(args.audio_path, args.offset, duration)
    if args.demucs or args.all:
        result['demucs'] = demucs_separate(args.audio_path)
    if args.whisper or args.all:
        result['whisper'] = whisper_lyrics(args.audio_path, args.offset, duration, model_size='base')
    if args.whisper_large:
        result['whisper'] = whisper_lyrics(args.audio_path, args.offset, duration, model_size='large')

    print(json.dumps(result, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder))


if __name__ == '__main__':
    main()
