import argparse
import json
import math
import os
import tempfile
from pathlib import Path
import sys

import librosa
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stt.inference import transcribe_audio


def chunk_audio_to_json(
    audio_path: str,
    output_path: str,
    chunk_sec: float = 30.0,
    overlap_sec: float = 2.0,
    target_sr: int = 16000,
):
    if overlap_sec >= chunk_sec:
        raise ValueError("overlap_sec must be smaller than chunk_sec")

    audio, sr = librosa.load(audio_path, sr=target_sr)
    duration = len(audio) / sr if sr else 0.0
    if duration <= 0:
        raise RuntimeError(f"No audio content found in {audio_path}")

    step_sec = chunk_sec - overlap_sec
    n_chunks = max(1, math.ceil((duration - overlap_sec) / step_sec))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows = []
    for chunk_id in range(n_chunks):
        start_sec = chunk_id * step_sec
        end_sec = min(start_sec + chunk_sec, duration)

        start_idx = int(start_sec * sr)
        end_idx = int(end_sec * sr)
        chunk_audio = audio[start_idx:end_idx]
        if len(chunk_audio) == 0:
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, chunk_audio, sr)
            text, confidence = transcribe_audio(
                tmp_path,
                decode="beam",
                beam_width=10,
                max_seconds=None,
                max_frames=None,
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        rows.append(
            {
                "chunk_id": chunk_id,
                "start_sec": round(start_sec, 4),
                "end_sec": round(end_sec, 4),
                "text": text,
                "confidence": round(float(confidence), 4),
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Convert audio to chunked JSON lines for RAG ingestion")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--output", required=True, help="Path to output .json/.jsonl file")
    parser.add_argument("--chunk-sec", type=float, default=30.0, help="Chunk duration in seconds")
    parser.add_argument("--overlap-sec", type=float, default=2.0, help="Overlap between chunks in seconds")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate")
    args = parser.parse_args()

    rows = chunk_audio_to_json(
        audio_path=args.audio,
        output_path=args.output,
        chunk_sec=args.chunk_sec,
        overlap_sec=args.overlap_sec,
        target_sr=args.sr,
    )

    print(f"[INFO] Wrote {len(rows)} chunks to {args.output}")
    if rows:
        print("[INFO] First chunk:")
        print(json.dumps(rows[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
