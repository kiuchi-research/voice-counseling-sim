#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from counseling_voice_demo.system_checks import find_ffmpeg


def main() -> None:
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path is None:
        print("ffmpeg が見つかりません。MP3結合を使う前に ffmpeg をインストールしてください。")
        sys.exit(1)

    print(f"ffmpeg found: {ffmpeg_path}")


if __name__ == "__main__":
    main()
