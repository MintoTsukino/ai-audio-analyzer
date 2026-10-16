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
# Basic analysis (whole file by default)
python analyze_audio.py song.wav

# Short Japanese summary for an AI to read and talk about
python analyze_audio.py song.wav --summary

# Specific modes
python analyze_audio.py song.wav --tags --bands --lufs

# Everything except Demucs and whisper-large
python analyze_audio.py song.wav --all

# Game BGM checks (silence, DC offset, loop point candidates on the beat grid)
python analyze_audio.py song.wav --game

# A/B comparison (loudness-matched tonal balance + number differences + overlay image A_vs_B.png)
python analyze_audio.py mine.wav --compare reference.wav --summary

# Mix doctor: rule-of-thumb findings (mud, harshness, low-end width, over-compression, true peak, mono, balance, DC)
python analyze_audio.py song.wav --doctor --summary

# Chord progression estimate (major/minor triads, 4/4 assumed, bar-by-bar, most repeated 4/8 bars)
python analyze_audio.py song.wav --chords --summary

# Render the sound as an image an AI can look at (loudness + clipping, mel spectrogram, harmonic/percussive, sections)
python analyze_audio.py song.wav --image                         # -> song_ears.png next to the file
python analyze_audio.py song.wav --image --image-from 100 --image-to 125   # zoom in (beats are drawn when <= 30 s)

# Analyze a specific section
python analyze_audio.py song.wav --offset 60 --duration 30

# High-precision lyrics (3GB model download on first use). Language is auto-detected unless --lang is given
python analyze_audio.py song.wav --whisper-large --lang ja
```

### What changed (2026-10)

- **Whole file by default.** Previously only the first 30 seconds were analyzed unless `--full` was given. Use `--duration` to limit. `--full` is kept for compatibility.
- **Bands, spectral features and peaks use the original sample rate (stereo).** Previously everything was resampled to 22.05 kHz mono, so content above ~11 kHz was missing from `brilliance_6000_20000hz`, and the peak was measured on the resampled mono signal.
- **Sample peak vs. true peak** (4x oversampled), plus a count of clipped samples.
- **Structure:** the number of sections follows the song length (about one per 20 s), boundaries snap to the nearest beat, and each section reports loudness, percussive ratio, low-end share and brightness. Also reports when percussion enters.
- **Beat grid fallback:** for pad/organ-heavy music where the beat tracker finds almost no beats, a grid is built from the estimated tempo and aligned to onsets.
- **PANNs tags cover the whole file** in 10-second windows (overall = max over windows) and include a per-window timeline. Model-loading messages no longer pollute the JSON on stdout.
- **Demucs** is launched with the current Python interpreter and stem energy is measured over the whole stem. `--all` no longer includes Demucs (too slow); pass `--demucs` explicitly.
- **Whisper** language is auto-detected (was hard-coded to Japanese).
- **`--summary`** prints a short Japanese summary instead of JSON. **`--game`** adds game-BGM checks.
- **`--compare`** matches loudness (250 Hz–4 kHz) before comparing tonal balance, so "louder sounds better" doesn't fool the comparison. Great for reference-track matching.
- **`--doctor`** flags bands that stick out from the song's own spectral tilt (fit over 100 Hz–4 kHz so dark/mp3-cut tops don't skew it), clipping, true peak above −1 dBTP, over-compression, low-end stereo width below 150 Hz, mono fold-down loss, L/R balance and DC offset. It does not check instrument masking.
- **`--chords`** uses the harmonic component's beat-synchronous chroma against major/minor templates with Viterbi smoothing, then groups beats into bars. On a synthetic test (Am–F–C–G ×4) it got 16/16 bars; with 7ths (Am7–Fmaj7–C–G7) 15/16. Tempos below 80 BPM are doubled (half-tempo misreads), so genuinely slow songs may be read at double speed.
- **`--image`** renders the sound as a picture (loudness with clipping marks, mel spectrogram up to the original Nyquist, harmonic vs. percussive energy, section boundaries). A multimodal AI can *look* at it and spot things numbers miss — filter sweeps, risers, where the hi-hats start, breakdowns.

### All flags

| Flag | What it does |
|---|---|
| `--deep` | MFCC, Tonnetz, spectral contrast, chroma |
| `--structure` | Sections (auto count, beat-snapped) with per-section loudness / percussion / low end / brightness, energy curve |
| `--rhythm` | Onset times, tempo stability |
| `--bands` | 7-band frequency energy distribution |
| `--hpss` | Harmonic / percussive ratio |
| `--pitch` | F0 tracking (monophonic) |
| `--tags` | PANNs ML tags (genre / instrument / mood), whole file in 10 s windows with a timeline |
| `--game` | Game BGM checks: leading/trailing silence, DC offset, loop point candidates |
| `--summary` | Short Japanese summary (enables structure, bands, hpss, tags, lufs, stereo, game) |
| `--lang XX` | Whisper language (default: auto-detect) |
| `--image [PATH]` | Render a PNG "ear image" for multimodal AIs (default: `<file>_ears.png`) |
| `--image-from N` / `--image-to N` | Time range of the image in seconds |
| `--compare B` | Compare with file B: numbers, loudness-matched 1/3-octave tonal curve, `A_vs_B.png` |
| `--doctor` | Mix doctor findings (relative to the song's own spectral tilt, not a genre average) |
| `--chords` | Chord progression estimate (triads only, 4/4 assumed; roughly 60-70% accurate on real music) |
| `--whisper` | Lyrics (whisper-base) |
| `--whisper-large` | Lyrics (whisper-large, 3GB model) |
| `--lufs` | LUFS loudness with platform targets |
| `--stereo` | Stereo image analysis |
| `--demucs` | Stem separation (slow) |
| `--all` | Everything except `--demucs` and `--whisper-large` |
| `--offset N` | Start offset in seconds (default 0) |
| `--duration N` | Analysis length in seconds (default: whole file) |
| `--full` | Analyze entire file (now the default; kept for compatibility) |

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
