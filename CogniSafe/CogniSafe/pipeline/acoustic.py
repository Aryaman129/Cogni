import subprocess
import csv
import os
import librosa
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────────────────────
# Update these paths to match your machine
OPENSMILE_EXE = os.path.join(os.path.dirname(__file__), '..', 'opensmile', 'bin', 'SMILExtract.exe')
EGEMAPS_CONF  = os.path.join(os.path.dirname(__file__), '..', 'opensmile', 'config', 'egemaps', 'v02', 'eGeMAPSv02.conf')
# ─────────────────────────────────────────────────────────────────────────────


def extract_acoustic_features(audio_path: str, transcript_data: dict = None) -> dict:
    """
    Extract acoustic biomarkers from an audio file.
    Returns a dict with 10 acoustic biomarker scores.
    """
    output_csv = audio_path.replace('.wav', '_smile.csv')

    # ── Step 1: Run openSMILE ────────────────────────────────────────────────
    cmd = [
        OPENSMILE_EXE,
        '-C', EGEMAPS_CONF,
        '-I', audio_path,
        '-csvoutput', output_csv,
        '-instname', 'test',
        '-appendcsv', '0'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"[openSMILE WARNING] {result.stderr[:300]}")
    except Exception as e:
        print(f"[openSMILE ERROR] {e}")
        return _fallback_acoustic()

    # ── Step 2: Parse the CSV output ─────────────────────────────────────────
    features = {}
    try:
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                features = row
                break  # only one row for file-level features
    except Exception as e:
        print(f"[CSV PARSE ERROR] {e}")
        return _fallback_acoustic()

    # ── Step 3: Extract the specific biomarkers we need ───────────────────────
    def safe_float(key, fallback=0.0):
        try:
            # eGeMAPS feature names — adjust if your version uses different names
            return float(features.get(key, fallback))
        except:
            return fallback

    pitch_mean  = safe_float('F0semitoneFrom27.5Hz_sma3nz_amean')
    pitch_range = safe_float('F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2')
    jitter      = safe_float('jitterLocal_sma3nz_amean')
    shimmer     = safe_float('shimmerLocaldB_sma3nz_amean')
    hnr         = safe_float('HNRdBACF_sma3nz_amean')

    # ── Step 4: Compute pause + speech rate using librosa ────────────────────
    pause_freq, pause_dur_mean, filled_pause_rate, speech_rate, articulation_rate = \
        compute_pause_and_rate(audio_path, transcript_data)

    acoustic = {
        'speech_rate':        round(speech_rate, 4),
        'articulation_rate':  round(articulation_rate, 4),
        'pause_frequency':    round(pause_freq, 4),
        'pause_duration_mean':round(pause_dur_mean, 4),
        'filled_pause_rate':  round(filled_pause_rate, 4),
        'pitch_mean':         round(pitch_mean, 4),
        'pitch_range':        round(pitch_range, 4),
        'jitter':             round(jitter, 6),
        'shimmer':            round(shimmer, 4),
        'HNR':                round(hnr, 4),
    }

    # cleanup temp file
    if os.path.exists(output_csv):
        os.remove(output_csv)

    return acoustic


def compute_pause_and_rate(audio_path: str, transcript_data: dict = None) -> tuple:
    """
    Compute pause frequency, mean pause duration, filled pause rate,
    speech rate, and articulation rate.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
    except Exception as e:
        print(f"[LIBROSA ERROR] {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Detect silent intervals (pauses)
    intervals = librosa.effects.split(y, top_db=30)  # non-silent intervals
    speech_duration = sum((end - start) for start, end in intervals) / sr

    # Pauses = gaps between non-silent intervals longer than 200ms
    pauses = []
    for i in range(1, len(intervals)):
        gap = (intervals[i][0] - intervals[i-1][1]) / sr
        if gap > 0.2:  # 200ms threshold
            pauses.append(gap)

    pause_freq     = len(pauses) / (duration / 60) if duration > 0 else 0  # pauses per minute
    pause_dur_mean = float(np.mean(pauses)) if pauses else 0.0

    # Speech rate = words / total duration (from transcript if available)
    word_count = 0
    if transcript_data and 'words' in transcript_data:
        word_count = len(transcript_data['words'])
    
    speech_rate       = (word_count / duration) * 60 if duration > 0 else 0  # words per minute
    articulation_rate = (word_count / speech_duration) * 60 if speech_duration > 0 else 0

    # Filled pauses (uh, um) — count from transcript if available
    filled_pauses = 0
    if transcript_data and 'text' in transcript_data:
        text = transcript_data['text'].lower()
        filled_pauses = text.count(' uh ') + text.count(' um ') + \
                        text.count(' uh,') + text.count(' um,')
    filled_pause_rate = (filled_pauses / (duration / 60)) if duration > 0 else 0

    return pause_freq, pause_dur_mean, filled_pause_rate, speech_rate, articulation_rate


def _fallback_acoustic() -> dict:
    """Returns default values if openSMILE fails — prevents pipeline crash."""
    return {
        'speech_rate': 0.0, 'articulation_rate': 0.0,
        'pause_frequency': 0.0, 'pause_duration_mean': 0.0,
        'filled_pause_rate': 0.0, 'pitch_mean': 0.0,
        'pitch_range': 0.0, 'jitter': 0.0,
        'shimmer': 0.0, 'HNR': 0.0,
    }