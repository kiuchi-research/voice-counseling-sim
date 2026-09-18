from __future__ import annotations

import base64
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from counseling_voice_demo.audio_probe import build_probe_report


DEV_AUDIO_PATH = ROOT_DIR / "app" / "assets" / "dev_probe_tone.mp3"


def _audio_data_url(audio_path: Path) -> str:
    encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    return f"data:audio/mpeg;base64,{encoded}"


def _player_html(audio_path: Path) -> str:
    audio_src = _audio_data_url(audio_path)
    return f"""
<!doctype html>
<html lang="ja">
  <head>
    <meta charset="utf-8" />
    <style>
      body {{
        font-family: sans-serif;
        margin: 0;
        color: #1f2933;
      }}
      .controls {{
        align-items: center;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 12px 0;
      }}
      button {{
        border: 1px solid #9aa5b1;
        border-radius: 6px;
        background: #ffffff;
        cursor: pointer;
        min-height: 36px;
        padding: 6px 10px;
      }}
      #log {{
        background: #f5f7fa;
        border: 1px solid #d9e2ec;
        border-radius: 6px;
        min-height: 90px;
        padding: 8px;
        white-space: pre-wrap;
      }}
      #position {{
        font-variant-numeric: tabular-nums;
      }}
    </style>
  </head>
  <body>
    <audio id="probe-audio" preload="auto" src="{audio_src}"></audio>
    <div class="controls">
      <button type="button" id="play">Play</button>
      <button type="button" id="pause">Pause</button>
      <button type="button" id="resume">Resume</button>
      <button type="button" id="skip">Skip</button>
      <span id="position">0.00s / 0.00s</span>
    </div>
    <div id="log"></div>
    <script>
      const audio = document.getElementById("probe-audio");
      const log = document.getElementById("log");
      const position = document.getElementById("position");

      function addLog(eventName) {{
        const row = `${{new Date().toISOString()}} ${{eventName}} current=${{audio.currentTime.toFixed(2)}} duration=${{Number.isFinite(audio.duration) ? audio.duration.toFixed(2) : "?"}}`;
        log.textContent = row + "\\n" + log.textContent;
      }}

      function updatePosition() {{
        const duration = Number.isFinite(audio.duration) ? audio.duration.toFixed(2) : "0.00";
        position.textContent = `${{audio.currentTime.toFixed(2)}}s / ${{duration}}s`;
      }}

      document.getElementById("play").addEventListener("click", () => {{
        audio.currentTime = 0;
        audio.play();
        addLog("play_clicked");
      }});
      document.getElementById("pause").addEventListener("click", () => {{
        audio.pause();
        addLog("pause_clicked");
      }});
      document.getElementById("resume").addEventListener("click", () => {{
        audio.play();
        addLog("resume_clicked");
      }});
      document.getElementById("skip").addEventListener("click", () => {{
        audio.pause();
        audio.currentTime = audio.duration || audio.currentTime;
        addLog("skip_clicked");
        updatePosition();
      }});

      ["loadedmetadata", "play", "pause", "ended", "timeupdate"].forEach((eventName) => {{
        audio.addEventListener(eventName, () => {{
          updatePosition();
          if (eventName !== "timeupdate") {{
            addLog(eventName);
          }}
        }});
      }});

      updatePosition();
    </script>
  </body>
</html>
"""


def main() -> None:
    st.set_page_config(
        page_title="Audio Player Probe",
        page_icon=None,
        layout="wide",
    )

    st.title("Audio Player Probe")
    st.caption("音声はAIにより生成されています。")

    report = build_probe_report(DEV_AUDIO_PATH)
    st.json(report)

    if not DEV_AUDIO_PATH.exists():
        st.error("固定MP3が見つかりません。`python utility/create_dev_probe_audio.py` を実行してください。")
        return

    components.html(_player_html(DEV_AUDIO_PATH), height=210)


if __name__ == "__main__":
    main()
