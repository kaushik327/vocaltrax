#!/usr/bin/env python3
"""Download a handful of LibriSpeech utterances for evaluation.

Usage:
    python fetch_librispeech.py                   # defaults: 8 utterances, 3-6s each
    python fetch_librispeech.py --n 16 --seed 7   # more utterances, different seed
    python fetch_librispeech.py --split test-clean # different split

Requires: soundfile, numpy, tqdm
Downloads the *dev-clean* split by default (~340 MB tar.gz), extracts a few
short utterances as 16-bit WAV into data/librispeech/, then cleans up.
"""

import argparse
import io
import os
import tarfile
import urllib.request
import shutil
import numpy as np
import soundfile

LIBRISPEECH_URL = {
    "dev-clean":  "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
}

SAMPLE_RATE = 16000  # LibriSpeech native SR


def collect_flac_members(tar: tarfile.TarFile):
    """Yield (member, speaker_id, chapter_id, utt_id) for every .flac in the archive."""
    for m in tar.getmembers():
        if m.name.endswith(".flac") and m.isfile():
            parts = os.path.splitext(os.path.basename(m.name))[0].split("-")
            if len(parts) >= 3:
                yield m, parts[0], parts[1], parts[2]


def main():
    parser = argparse.ArgumentParser(description="Fetch LibriSpeech utterances")
    parser.add_argument("--split", default="dev-clean", choices=list(LIBRISPEECH_URL))
    parser.add_argument("--n", type=int, default=8, help="Number of utterances to keep")
    parser.add_argument("--min-dur", type=float, default=3.0, help="Min duration (s)")
    parser.add_argument("--max-dur", type=float, default=6.0, help="Max duration (s)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None, help="Output directory (default: data/librispeech)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "data", "librispeech")
    os.makedirs(out_dir, exist_ok=True)

    url = LIBRISPEECH_URL[args.split]
    tar_path = os.path.join(out_dir, f"{args.split}.tar.gz")

    # Download
    if not os.path.exists(tar_path):
        print(f"Downloading {args.split} from {url} ...")
        urllib.request.urlretrieve(url, tar_path)
        print("Done.")
    else:
        print(f"Using cached {tar_path}")

    # Scan archive for suitable utterances
    print("Scanning archive for utterances ...")
    rng = np.random.default_rng(args.seed)
    candidates = []

    with tarfile.open(tar_path, "r:gz") as tar:
        for member, spk, chap, utt in collect_flac_members(tar):
            # Quick duration estimate from file size (FLAC ~50% of raw 16-bit)
            est_samples = member.size * 2  # rough
            est_dur = est_samples / SAMPLE_RATE
            if args.min_dur <= est_dur <= args.max_dur * 3:  # loose filter; refine after decode
                candidates.append(member)

        print(f"  {len(candidates)} candidates found (loose filter)")
        rng.shuffle(candidates)

        kept = 0
        for member in candidates:
            if kept >= args.n:
                break
            f = tar.extractfile(member)
            if f is None:
                continue
            audio, sr = soundfile.read(io.BytesIO(f.read()))
            dur = len(audio) / sr
            if dur < args.min_dur or dur > args.max_dur:
                continue

            basename = os.path.splitext(os.path.basename(member.name))[0]
            wav_path = os.path.join(out_dir, f"{basename}.wav")
            soundfile.write(wav_path, audio, sr, subtype="PCM_16")
            print(f"  [{kept+1}/{args.n}] {basename}.wav  ({dur:.1f}s)")
            kept += 1

    if kept < args.n:
        print(f"Warning: only found {kept}/{args.n} utterances matching duration filter")

    # Clean up tarball to save space
    if os.path.exists(tar_path):
        os.remove(tar_path)
        print(f"Removed {tar_path}")

    print(f"\n{kept} utterances saved to {out_dir}/")
    print(f"Run with: python synthesize.py general.target=data/librispeech/<name>.wav")


if __name__ == "__main__":
    main()
