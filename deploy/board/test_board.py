#!/usr/bin/env python3
"""Board-side smoke test: run the axmodel on the bundled test wavs and report timings."""

import argparse
import json
from pathlib import Path

from asr_engine import Asr, read_wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="/root/x-asr/models")
    ap.add_argument("--wav-dir", default="/root/x-asr/test_wavs")
    args = ap.parse_args()

    asr = Asr(args.models_dir)
    print("model io:", json.dumps(asr.io, indent=1), flush=True)

    for wav in sorted(Path(args.wav_dir).glob("*.wav")):
        pcm = read_wav(wav)
        r = asr.transcribe(pcm)
        print(f"{wav.name}  {r['audio_ms']/1000:.2f}s  "
              f"enc={r['encoder_ms']:.0f}ms dec={r['decoder_joiner_ms']:.0f}ms "
              f"total={r['total_ms']:.0f}ms rtf={r['rtf']}  | {r['text']}", flush=True)


if __name__ == "__main__":
    main()
