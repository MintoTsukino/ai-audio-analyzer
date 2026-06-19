# ai-audio-analyzer

> Give your AI assistant ears.

A Python script that lets AI assistants (Claude Code, GPT, etc.) "listen" to music by extracting structured audio analysis data — tempo, key, frequency bands, genre tags, lyrics, loudness, stereo image, and more.

AIアシスタント（Claude Code、GPT 等）に「耳」をつけるための Python スクリプト。テンポ・キー・周波数帯・ジャンルタグ・歌詞・ラウドネス・ステレオ感などを数値化して返します。

## なぜこれが必要か / Why

AI と一緒に音楽を作りたいけど、AI は音を聴けない。
歌詞は書けるしコードも書けるけど、実際の音の判断はあなた任せ。
それじゃあ「一緒に作る」って言えない。

数値で見る「目」を持たせて、AI が音楽の話に参加できるようにします。

> "I want to collaborate with AI on music, but AI can't hear. Lyrics, structure, code — fine. But actual sound judgment falls on me alone. That isn't collaboration."
> This script gives AI a numerical "eye" to look at sound — not a perfect ear, but enough to participate in musical conversations.

## できること / Features

- 🎵 **Basic** — Tempo (BPM), key estimation (Krumhansl-Schmuckler), spectral features, peak/RMS dB
- 🎼 **Deep** — MFCC (timbre fingerprint), Tonnetz (harmonic), spectral contrast, chroma distribution
- 🏗️ **Structure** — Section boundaries, energy curve over time, onset strength
- 🥁 **Rhythm** — Onset detection, tempo stability
- 🎚️ **Bands** — Energy distribution across 7 frequency bands (Sub-Bass → Brilliance)
- 🎭 **HPSS** — Harmonic / percussive separation ratio
- 🎤 **Pitch** — F0 tracking (monophonic)
- 🤖 **Tags** — PANNs ML-based genre / instrument / mood tagging
- 📝 **Whisper** — Lyrics transcription (base / large)
- 📊 **LUFS** — Professional loudness measurement with platform targets
- 🎧 **Stereo** — L/R correlation, stereo width, phase
- 🎸 **Demucs** — Stem separation (vocals / drums / bass / other)

## Install

```bash
pip install librosa pyloudnorm panns_inference openai-whisper demucs "numpy<2"
```

PANNs requires its model checkpoint and labels CSV; the first run will download them automatically (or place them at `~/panns_data/` manually if behind a firewall).

Whisper downloads its model on first use (~150MB for base, ~3GB for large).

## Usage

```bash
# Basic analysis (first 30 seconds)
python analyze_audio.py song.wav

# Specific modes
python analyze_audio.py song.wav --tags --bands --lufs

# Full analysis (heavy)
python analyze_audio.py song.wav --all

# Analyze a specific section
python analyze_audio.py song.wav --offset 60 --duration 30

# Whole file
python analyze_audio.py song.wav --full

# High-precision lyrics (3GB model download on first use)
python analyze_audio.py song.wav --whisper-large
```

### All flags

| Flag | What it does |
|---|---|
| `--deep` | MFCC, Tonnetz, spectral contrast, chroma |
| `--structure` | Section boundaries, energy curve |
| `--rhythm` | Onset times, tempo stability |
| `--bands` | 7-band frequency energy distribution |
| `--hpss` | Harmonic / percussive ratio |
| `--pitch` | F0 tracking (monophonic) |
| `--tags` | PANNs ML tags (genre / instrument / mood) |
| `--whisper` | Lyrics (whisper-base) |
| `--whisper-large` | Lyrics (whisper-large, 3GB model) |
| `--lufs` | LUFS loudness with platform targets |
| `--stereo` | Stereo image analysis |
| `--demucs` | Stem separation (slow) |
| `--all` | Everything except `--whisper-large` |
| `--offset N` | Start offset in seconds (default 0) |
| `--duration N` | Analysis length in seconds (default 30) |
| `--full` | Analyze entire file (slow) |

## Output

JSON to stdout. Pipe to a file:

```bash
python analyze_audio.py song.wav --all > analysis.json
```

See `examples/sample_output.json` for an actual analysis result.

## Limitations

This isn't a replacement for human ears.

- ❌ Cannot judge "groove", "emotion", or aesthetic quality
- ❌ Cannot evaluate mix "feel"
- ❌ Whisper struggles with shouted / distorted / screamed vocals (punk, metal, hardcore)
- ❌ `whisper-large` can hallucinate on out-of-distribution audio (we saw it return `"作詞・作曲・編曲 初音ミク"` × 5 on a Japanese punk track)
- ❌ Genre tags use AudioSet's 527 labels — no "Punk" tag exists, so punk tracks register as "Rock music"

What it gives you is a **numerical view** of audio that an AI can reason about — enough to participate in musical discussion, not enough to replace your judgment.

## Notes on dependencies

- `numpy<2` is pinned because PyTorch-based libraries (Whisper, PANNs, Demucs) haven't fully migrated to NumPy 2.x as of 2026.
- CPU inference is fine for typical use. GPU only helps Demucs noticeably.
- AMD GPUs on Windows: Don't bother with ROCm / DirectML — the CPU path is simpler and fast enough.

## License

MIT. Use it however you like.

## See also

- Companion note article: "AIに耳をつけてみた" (Japanese)
- [librosa](https://librosa.org/)
- [openai-whisper](https://github.com/openai/whisper)
- [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn)
- [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm)
- [Demucs](https://github.com/facebookresearch/demucs)
