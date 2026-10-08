"""Audio-only features for the tone model C. One feature vector per SEGMENT (a labelled recording's active
region, or one executed note of a take). Windows of 0.5 s (hop 0.25 s) inside the segment, each window ->
the classifier repo's 40 hand-engineered scalars (HNR, flatness, F0 stability, envelope, attack, MFCC stats)
+ 64 log-mel band means; aggregated over windows as mean and std. No physical features, no classifier.
Audio is peak-normalised per SEGMENT before feature extraction (level-invariant model; level is reported
separately by the acoustic diagnostics). Revision: 2026-09-06.
CELLO_REPO_ROOT defaults to this checkout; CELLO_DATA_ROOT selects local
perform_* recordings. Neither setting changes feature extraction."""
import os, sys, json
import numpy as np, soundfile as sf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_paths import repo_root, data_root, require_file, require_directory
REPO = str(repo_root())
DATA_ROOT = str(data_root())
require_file(Path(REPO) / "SoundClassifier" / "audio_features.py", "CELLO_REPO_ROOT")
for _p in (REPO, os.path.join(REPO, 'SoundClassifier')):
    if _p not in sys.path: sys.path.insert(0, _p)   # dataset.py does a bare `import audio_features`
from SoundClassifier import audio_features as af
SR = af.SR; WIN_S = 0.5; HOP_S = 0.25
VERSION = 'C-feat-v1'

def _windows(seg, sr=SR):
    n = int(round(WIN_S * sr)); h = int(round(HOP_S * sr))
    if len(seg) <= n:
        chunk = np.pad(seg, (0, n - len(seg)), mode='reflect') if len(seg) > 1 else np.zeros(n, np.float32)
        return [chunk.astype(np.float32)]
    out = []; pos = 0
    while pos + n <= len(seg): out.append(seg[pos:pos + n].astype(np.float32)); pos += h
    if out and pos - h + n < len(seg): out.append(seg[len(seg) - n:].astype(np.float32))
    return out

def _window_vec(chunk, sr=SR):
    scal = np.asarray(af.extract_scalar_features(chunk, sr), dtype=np.float64)
    mel = af.compute_log_mel_spectrogram(chunk, sr)            # (64, frames)
    return np.concatenate([scal, mel.mean(axis=1)])

def feature_names():
    n_scal = len(af.extract_scalar_features(np.random.RandomState(0).randn(int(WIN_S * SR)).astype(np.float32) * 0.1, SR))
    base = [f'scalar_{i}' for i in range(n_scal)] + [f'mel_{i}' for i in range(af.N_MELS)]
    return [f'{b}_mean' for b in base] + [f'{b}_std' for b in base] + ['n_windows', 'dur_s']

def segment_features(seg, sr=SR):
    seg = np.asarray(seg, dtype=np.float32)
    if seg.ndim > 1: seg = seg.mean(axis=1)
    seg = af.normalize_peak(seg)
    W = np.stack([_window_vec(c, sr) for c in _windows(seg, sr)])
    return np.concatenate([W.mean(axis=0), W.std(axis=0), [W.shape[0], len(seg) / sr]])

def load_audio(path):
    path = require_file(path, "audio input / CELLO_DATA_ROOT")
    y, sr = sf.read(path, dtype='float32', always_2d=True); y = y.mean(axis=1)
    if sr != SR:
        import librosa; y = librosa.resample(y, orig_sr=sr, target_sr=SR); sr = SR
    return y, sr

def recording_segment(path, record=None):
    """Active region of a labelled recording (dataset.py's _active_region: audio_timing with pads)."""
    y, sr = load_audio(path)
    if record is not None:
        from SoundClassifier import dataset as ds
        a, b = ds._active_region(record, len(y), sr); y = y[int(a * sr):int(b * sr)]
    return y, sr

def take_note_segments(perform_dir):
    """Executed notes of a take, sliced exactly as rl/posthoc_score.py does (lead + actual_onset, actual_duration)."""
    d = require_directory(perform_dir, "CELLO_DATA_ROOT (take directory)")
    wavs = [f for f in os.listdir(d) if f.endswith('.wav')]
    if not wavs:
        raise FileNotFoundError(f"No WAV in {d}; check CELLO_DATA_ROOT.")
    wav = wavs[0]
    require_file(d / (wav[:-4] + '_performance.json'), "CELLO_DATA_ROOT (performance metadata)")
    y, sr = load_audio(os.path.join(d, wav)); perf = json.load(open(os.path.join(d, wav[:-4] + '_performance.json')))
    lead = float(perf.get('audio_lead_in_s', 0.0)); out = []
    for t in perf['timeline']:
        on, dur = float(t['actual_onset']), float(t['actual_duration']); lo = int((lead + on) * sr)
        seg = y[lo:lo + int(dur * sr)]
        if len(seg) < sr // 20: continue
        out.append(dict(note_index=t.get('note_index'), dur_s=dur, dynamic=t.get('dynamic'), audio=seg, sr=sr))
    return out
