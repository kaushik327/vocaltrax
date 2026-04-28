import numpy as np
import parselmouth
from parselmouth.praat import call


# Peterson & Barney (1952) mean F1/F2 values (Hz) for American English vowels.
# Source: Peterson & Barney, "Control methods used in a study of the vowels,"
# JASA 24(2), 1952, Table II — adult male averages.
PETERSON_BARNEY_VOWELS = {
    "iː":  {"f1": 270,  "f2": 2290},  # heed
    "ɪ":   {"f1": 390,  "f2": 1990},  # hid
    "eɪ":  {"f1": 530,  "f2": 1840},  # head  (mapped to /ɛ/)
    "ɛ":   {"f1": 530,  "f2": 1840},  # head
    "æ":   {"f1": 660,  "f2": 1720},  # had
    "ɑː":  {"f1": 730,  "f2": 1090},  # hod
    "ɔː":  {"f1": 570,  "f2": 840},   # hawed
    "ʊ":   {"f1": 440,  "f2": 1020},  # hood
    "uː":  {"f1": 300,  "f2": 870},   # who'd
    "ʌ":   {"f1": 640,  "f2": 1190},  # hud
    "ɝː":  {"f1": 490,  "f2": 1350},  # heard
}


def extract_formants(audio, sr, max_formant=5500.0, num_formants=5,
                     window_length=0.025, time_step=0.01,
                     pre_emphasis_from=50.0):
    """Extract F1 and F2 time-series from an audio signal using Praat's Burg method.

    Args:
        audio: 1-D numpy array of audio samples.
        sr: Sample rate in Hz.
        max_formant: Maximum formant frequency (Hz). 5500 is standard for male
            voices; use ~5500 for male, ~6500 for female.
        num_formants: Number of formants to track (Praat default: 5).
        window_length: Analysis window length in seconds.
        time_step: Time step between formant estimates in seconds.
        pre_emphasis_from: Pre-emphasis frequency (Hz).

    Returns:
        dict with keys:
            "times"  — 1-D array of time stamps (seconds)
            "f1"     — 1-D array of F1 values (Hz), 0 where undefined
            "f2"     — 1-D array of F2 values (Hz), 0 where undefined
            "f3"     — 1-D array of F3 values (Hz), 0 where undefined
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    formant_obj = call(snd, "To Formant (burg)",
                       time_step, num_formants, max_formant,
                       window_length, pre_emphasis_from)

    n_frames = call(formant_obj, "Get number of frames")
    times = np.array([call(formant_obj, "Get time from frame number", i + 1)
                      for i in range(n_frames)])
    f1 = np.array([call(formant_obj, "Get value at time", 1, t, "Hertz", "Linear")
                   for t in times])
    f2 = np.array([call(formant_obj, "Get value at time", 2, t, "Hertz", "Linear")
                   for t in times])
    f3 = np.array([call(formant_obj, "Get value at time", 3, t, "Hertz", "Linear")
                   for t in times])

    # Praat returns NaN for undefined formants; replace with 0
    f1 = np.nan_to_num(f1, nan=0.0)
    f2 = np.nan_to_num(f2, nan=0.0)
    f3 = np.nan_to_num(f3, nan=0.0)

    return {"times": times, "f1": f1, "f2": f2, "f3": f3}


def formant_trajectory_error(target_formants, resynth_formants):
    """Compute per-frame and aggregate formant trajectory errors.

    Both inputs must be dicts as returned by `extract_formants`.  The shorter
    array is used as the comparison length (they should normally be equal).

    Returns:
        dict with keys:
            "f1_rmse"      — RMSE of F1 trajectory (Hz)
            "f2_rmse"      — RMSE of F2 trajectory (Hz)
            "f1f2_rmse"    — joint RMSE in (F1, F2) space (Hz)
            "f1_corr"      — Pearson correlation of F1 trajectories
            "f2_corr"      — Pearson correlation of F2 trajectories
            "n_frames"     — number of frames compared
    """
    n = min(len(target_formants["f1"]), len(resynth_formants["f1"]))
    tf1, tf2 = target_formants["f1"][:n], target_formants["f2"][:n]
    rf1, rf2 = resynth_formants["f1"][:n], resynth_formants["f2"][:n]

    # Mask frames where both target and resynth have valid formants
    valid = (tf1 > 0) & (rf1 > 0) & (tf2 > 0) & (rf2 > 0)
    if valid.sum() == 0:
        return {"f1_rmse": np.nan, "f2_rmse": np.nan, "f1f2_rmse": np.nan,
                "f1_corr": np.nan, "f2_corr": np.nan, "n_frames": 0}

    tf1v, tf2v = tf1[valid], tf2[valid]
    rf1v, rf2v = rf1[valid], rf2[valid]

    f1_rmse = np.sqrt(np.mean((tf1v - rf1v) ** 2))
    f2_rmse = np.sqrt(np.mean((tf2v - rf2v) ** 2))
    f1f2_rmse = np.sqrt(np.mean((tf1v - rf1v) ** 2 + (tf2v - rf2v) ** 2))

    def _pearson(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return np.nan
        return np.corrcoef(a, b)[0, 1]

    return {
        "f1_rmse": float(f1_rmse),
        "f2_rmse": float(f2_rmse),
        "f1f2_rmse": float(f1f2_rmse),
        "f1_corr": float(_pearson(tf1v, rf1v)),
        "f2_corr": float(_pearson(tf2v, rf2v)),
        "n_frames": int(valid.sum()),
    }


def nearest_vowel(f1, f2, vowel_table=None):
    """Return the IPA label of the nearest Peterson & Barney vowel to (f1, f2).

    Args:
        f1: F1 value in Hz.
        f2: F2 value in Hz.
        vowel_table: Optional dict mapping IPA labels to {"f1": ..., "f2": ...}.
            Defaults to PETERSON_BARNEY_VOWELS.

    Returns:
        (label, distance) tuple.
    """
    if vowel_table is None:
        vowel_table = PETERSON_BARNEY_VOWELS
    best_label, best_dist = None, np.inf
    for label, coords in vowel_table.items():
        d = np.sqrt((f1 - coords["f1"]) ** 2 + (f2 - coords["f2"]) ** 2)
        if d < best_dist:
            best_label, best_dist = label, d
    return best_label, float(best_dist)
