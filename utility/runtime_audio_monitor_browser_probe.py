from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
COMPONENT_PATH = ROOT_DIR / "app" / "components" / "runtime_audio_monitor" / "index.html"
DEFAULT_CHROMIUM = "/snap/bin/chromium"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe the runtime audio monitor component in a headless browser.",
    )
    parser.add_argument("--chromium", default=DEFAULT_CHROMIUM)
    parser.add_argument("--keep-html", action="store_true")
    args = parser.parse_args(argv)

    html = build_probe_html(COMPONENT_PATH.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(
        prefix=".runtime-audio-monitor-probe.",
        dir=ROOT_DIR,
    ) as tmp_dir:
        probe_path = Path(tmp_dir) / "probe.html"
        probe_path.write_text(html, encoding="utf-8")
        xdg_runtime_dir = Path(tmp_dir) / "xdg-runtime"
        xdg_runtime_dir.mkdir(mode=0o700)
        browser_env = dict(os.environ)
        browser_env["XDG_RUNTIME_DIR"] = str(xdg_runtime_dir)
        command = [
            args.chromium,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--autoplay-policy=no-user-gesture-required",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=2500",
            "--dump-dom",
            probe_path.as_uri(),
        ]
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            env=browser_env,
        )
        if args.keep_html:
            kept_path = Path.cwd() / "runtime_audio_monitor_probe.html"
            kept_path.write_text(html, encoding="utf-8")
            print(f"kept_html={kept_path}")
        if completed.returncode != 0:
            print(completed.stderr, file=sys.stderr)
            return completed.returncode
        result = parse_probe_result(completed.stdout)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        if result.get("audioRunning") is not True:
            return 1
        if result.get("timelineHasTranscript") is not True:
            return 1
        if "playback_ended" in completed.stdout:
            print("component emitted playback_ended unexpectedly", file=sys.stderr)
            return 1
        return 0


def build_probe_html(component_html: str) -> str:
    args = {
        "ws_url": "ws://127.0.0.1:8765/runtime/monitor-audio",
        "connect": False,
        "volume": 1.0,
        "muted": False,
        "start_url": "http://127.0.0.1:8765/runtime/start",
        "start_options": {"max_turns": 2},
    }
    head_script = f"""
    <script>
      window.RUNTIME_AUDIO_MONITOR_ARGS = {json.dumps(args)};
      window.__probeAudioSourceStartCount = 0;
      function probePatchAudioContext(AudioContextClass) {{
        if (!AudioContextClass || AudioContextClass.prototype.__probePatched) {{
          return;
        }}
        const originalCreateBufferSource = AudioContextClass.prototype.createBufferSource;
        AudioContextClass.prototype.createBufferSource = function(...sourceArgs) {{
          const source = originalCreateBufferSource.apply(this, sourceArgs);
          const originalStart = source.start;
          source.start = function(...startArgs) {{
            window.__probeAudioSourceStartCount += 1;
            return originalStart.apply(this, startArgs);
          }};
          return source;
        }};
        AudioContextClass.prototype.__probePatched = true;
      }}
      probePatchAudioContext(window.AudioContext);
      probePatchAudioContext(window.webkitAudioContext);
      function probeZeroPcmBase64(byteLength) {{
        let binary = "";
        for (let index = 0; index < byteLength; index += 1) {{
          binary += String.fromCharCode(0);
        }}
        return btoa(binary);
      }}
      class ProbeWebSocket extends EventTarget {{
        constructor(url) {{
          super();
          this.url = url;
          this.readyState = ProbeWebSocket.CONNECTING;
          queueMicrotask(() => {{
            this.readyState = ProbeWebSocket.OPEN;
            this.dispatchEvent(new Event("open"));
            queueMicrotask(() => {{
              this.dispatchEvent(new MessageEvent("message", {{
                data: JSON.stringify({{
                  type: "transcript_final",
                  text: "probe transcript",
                  metadata: {{
                    session_id: "probe-session",
                    turn_id: 1,
                    speaker: "counselor",
                    speaker_id: "counselor",
                    display_name: "Probe counselor",
                    role: "counselor",
                    segment_id: "probe-session:1:counselor"
                  }}
                }})
              }}));
            }});
            queueMicrotask(() => {{
              this.dispatchEvent(new MessageEvent("message", {{
                data: JSON.stringify({{
                  type: "audio_chunk",
                  pcm_base64: probeZeroPcmBase64(4800),
                  metadata: {{
                    session_id: "probe-session",
                    turn_id: 1,
                    speaker: "counselor",
                    chunk_index: 0,
                    sample_rate: 24000,
                    sample_width_bits: 16,
                    channels: 1,
                    duration_ms: 100
                  }}
                }})
              }}));
            }});
            queueMicrotask(() => {{
              this.dispatchEvent(new MessageEvent("message", {{
                data: JSON.stringify({{type: "stream_end"}})
              }}));
            }});
          }});
        }}
        close() {{
          this.readyState = ProbeWebSocket.CLOSED;
          this.dispatchEvent(new CloseEvent("close", {{code: 1000}}));
        }}
      }}
      ProbeWebSocket.CONNECTING = 0;
      ProbeWebSocket.OPEN = 1;
      ProbeWebSocket.CLOSING = 2;
      ProbeWebSocket.CLOSED = 3;
      window.WebSocket = ProbeWebSocket;
      window.fetch = async () => new Response(
        JSON.stringify({{session_id: "probe-session", phase: "running"}}),
        {{status: 200, headers: {{"Content-Type": "application/json"}}}}
      );
    </script>
    """
    observer_script = """
    <script>
      (async () => {
        window.__runtimeAudioMonitorProbe = {statusLog: [], audioRunning: false};
        const status = document.getElementById("status");
        const button = document.getElementById("start-runtime");
        const record = () => {
          const value = status ? status.textContent : "";
          window.__runtimeAudioMonitorProbe.statusLog.push(value);
          if (value === "audio_context_running") {
            window.__runtimeAudioMonitorProbe.audioRunning = true;
          }
          document.body.setAttribute(
            "data-status-log",
            window.__runtimeAudioMonitorProbe.statusLog.join("|")
          );
          document.body.setAttribute(
            "data-audio-running",
            window.__runtimeAudioMonitorProbe.audioRunning ? "true" : "false"
          );
        };
        if (status) {
          new MutationObserver(record).observe(status, {childList: true});
          record();
        }
        if (button) {
          button.click();
        }
        for (let index = 0; index < 100; index += 1) {
          await Promise.resolve();
        }
        await new Promise((resolve) => setTimeout(resolve, 1200));
        const timelineText = (
          document.getElementById("timeline-lanes")?.textContent || ""
        ).trim();
        const result = document.createElement("pre");
        result.id = "probe-result";
        const audioSourceStarted = Number(window.__probeAudioSourceStartCount || 0) > 0;
        result.textContent = JSON.stringify({
          status: status ? status.textContent : "",
          audioRunning: window.__runtimeAudioMonitorProbe.audioRunning || audioSourceStarted,
          audioSourceStartCount: Number(window.__probeAudioSourceStartCount || 0),
          statusLog: window.__runtimeAudioMonitorProbe.statusLog,
          timelineText,
          timelineHasTranscript: timelineText.includes("probe transcript")
        });
        document.body.appendChild(result);
      })();
    </script>
    """
    html = component_html.replace("</head>", f"{head_script}\n  </head>")
    return html.replace("</body>", f"{observer_script}\n  </body>")


def parse_probe_result(dom: str) -> dict[str, object]:
    match = re.search(r'<pre id="probe-result">([^<]+)</pre>', dom)
    if match is not None:
        return json.loads(match.group(1))
    body_match = re.search(
        r'<body[^>]*data-status-log="([^"]*)"[^>]*data-audio-running="([^"]*)"',
        dom,
    )
    if body_match is None:
        raise RuntimeError("probe result was not written to the dumped DOM")
    return {
        "status": "",
        "audioRunning": body_match.group(2) == "true",
        "statusLog": body_match.group(1).split("|"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
