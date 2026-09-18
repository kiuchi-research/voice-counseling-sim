#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT_DIR / "app" / "assets" / "dev_probe_tone.mp3"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=880:duration=1.2",
        "-q:a",
        "9",
        str(OUTPUT_PATH),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        print("ffmpeg が見つかりません。固定MP3を作る前に ffmpeg をインストールしてください。")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr)
        sys.exit(exc.returncode)

    print(f"created: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
