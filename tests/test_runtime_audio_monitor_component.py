from __future__ import annotations

from pathlib import Path


COMPONENT_PATH = Path("app/components/runtime_audio_monitor/index.html")


def test_runtime_audio_monitor_component_uses_web_audio_and_websocket() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "new WebSocket(args.ws_url)" in html
    assert "window.fetch(args.start_url" in html
    assert "AudioContext" in html
    assert "createBuffer" in html
    assert "pcm_base64" in html
    assert "sample_width_bits" in html


def test_runtime_audio_monitor_component_requires_component_click_for_audio_start() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "Connect & Resume audio" not in html
    assert '<button id="connect"' not in html
    assert '<button id="disconnect"' not in html
    assert "Connect & Start audio" in html
    assert '<button id="pause-runtime" type="button">Pause</button>' in html
    assert '<button id="resume-runtime" type="button">Resume</button>' in html
    assert '<button id="recover-runtime" type="button">Refresh & Reconnect</button>' in html
    assert "click Connect & Start audio" in html
    assert "context.resume()" in html
    assert "userGesture: false" in html
    assert "audioContext.state !== \"running\"" in html
    assert "startRuntimeWithAudio" in html
    assert "const connected = await connectMonitor({" in html
    assert "resetTimelineOnOpen: !alreadyConnected" in html
    assert "HUMAN_INITIAL_INPUT_PREROLL_MS" in html
    assert "beginHumanInitialInputPreRollCapture" in html
    assert "cancelHumanInitialInputPreRollCapture" in html
    start_index = html.index("async function startRuntimeWithAudio()")
    start_body = html[
        start_index : html.index("async function postRuntimeControl", start_index)
    ]
    assert start_body.index("beginHumanInitialInputPreRollCapture()") < start_body.index(
        "await connectMonitor({"
    )
    assert start_body.index("await ensureHumanMicrophoneEnabled()") < start_body.index(
        "window.fetch(args.start_url"
    )
    assert start_body.index("await startAlwaysBargeInCaptureMonitor()") < start_body.index(
        "window.fetch(args.start_url"
    )


def test_runtime_audio_monitor_component_pauses_and_resumes_audio_playback() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "function pauseAudioPlayback()" in html
    assert "function resumeAudioPlayback()" in html
    assert "await audioContext.suspend()" in html
    assert "await audioContext.resume()" in html
    assert "function canScheduleAudio()" in html
    assert "if (!canScheduleAudio())" in html
    assert "audio paused" in html
    assert "audio resumed" in html
    assert "audioPlaybackPaused" in html
    assert "pausedPlaybackMessages" in html
    assert "function queuePausedPlaybackMessage(message)" in html
    assert "function flushPausedPlaybackMessages()" in html
    assert "if (audioPlaybackPaused)" in html
    assert "runtime_pause_url" in html
    assert "function postRuntimeControl(url, failurePrefix, payload)" in html
    assert "pauseRuntimeGenerationForPlayback" in html
    assert "resumeRuntimeGenerationForPlayback" in html
    assert "window.fetch(url" in html
    assert "body: JSON.stringify(bodyPayload)" in html
    assert 'reason: "playback"' in html
    assert "missing url" in html
    assert "audio paused; runtime pause not confirmed" in html
    assert "audio resumed; runtime resume not confirmed" in html
    assert "runtimeStartRequested" in html


def test_runtime_audio_monitor_component_buffers_initial_warmup_playback() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "先行生成中" in html
    assert '<div class="warmup-indicator" id="warmup-indicator" aria-live="polite">' in html
    assert "warmup_target_turns: 2" in html
    assert "warmup_min_chunks: 10" in html
    assert "warmup_max_wait_ms: 24000" in html
    assert "let warmupActive = false" in html
    assert "let warmupBufferedMessages = []" in html
    assert "const warmupCompletedTurnKeys = new Set()" in html
    assert "function beginWarmupBuffering()" in html
    assert "function maybeBufferWarmupMessage(message)" in html
    assert "function flushWarmupPlaybackBuffer(reason)" in html
    assert "if (warmupReady())" in html
    assert "targetTurns <= 0 && minChunks > 0" in html
    assert "resetPlaybackLaneSchedules(audioContext.currentTime)" in html
    assert "beginWarmupBuffering();" in html
    assert "cancelWarmupBuffering();" in html


def test_runtime_audio_monitor_component_adds_gap_between_playback_turns() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "playback_inter_turn_gap_min_ms: 900" in html
    assert "playback_inter_turn_gap_max_ms: 1600" in html
    assert "function playbackInterTurnGapMinMs()" in html
    assert "function playbackInterTurnGapMaxMs()" in html
    assert "function playbackInterTurnGapSeconds()" in html
    assert "Math.random()" in html
    assert "Math.floor(minMs + Math.random() * (maxMs - minMs + 1))" in html
    assert 'lastPlaybackKey: ""' in html
    assert "lane.lastPlaybackKey && lane.lastPlaybackKey !== playbackKey" in html
    assert "context.currentTime + PLAYBACK_LEAD_SECONDS + interTurnGapSeconds" in html
    assert "lane.scheduledTime + interTurnGapSeconds" in html
    assert "lane.lastPlaybackKey = playbackKey" in html


def test_runtime_audio_monitor_component_recovers_generation_pause_by_reconnect() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "runtime_stop_url" in html
    assert "function stopRuntimeForRecovery()" in html
    assert "postRuntimeControl(args.runtime_stop_url" in html
    assert "runtime stop failed" in html
    assert "function refreshAndReconnectRuntimeMonitor()" in html
    assert "status: \"refreshing runtime\"" in html
    assert "clearArmed: true" in html
    assert "const stopped = await stopRuntimeForRecovery()" in html
    assert "refreshed; click Connect & Start audio" in html
    assert "runtimeStartRequested = false" in html


def test_runtime_audio_monitor_component_shows_inline_runtime_wait_state() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert 'id="runtime-inline-state"' in html
    assert "function runtimeInlineStateForStatus(status)" in html
    assert "pause_reason: playback / 音声再生待ち: Resumeを押す" in html
    assert "humanSpeakerDisplayName(humanSpeakerId(status))" in html
    assert "const waitingForHumanInput = runtimeStatusHumanInputWait(status);" in html
    assert "waitingForHumanInput &&" in html
    assert "runtimeStatusAllowsHumanRecording(status, {" in html
    assert "function runtimeStatusShowsHumanCounselorInput(status)" in html
    assert 'humanInputState === "streaming_human_audio"' in html
    assert 'humanInputState === "streaming_human_audio_transcribing"' in html
    assert (
        'setRuntimeInlineState(`${humanSpeakerDisplayName(humanSpeakerId())}入力待ち`, "");'
        in html
    )
    assert "activeSources.size === 0" in html
    assert "function syncRuntimeInlineStateFromMonitorMetadata(metadata)" in html
    assert 'setRuntimeInlineState("", "");' in html
    assert "active_speaker_id: speaker" in html
    assert "!warmupActive" in html
    assert "startRuntimeInlineStatusPolling" in html
    assert "window.setInterval(refreshRuntimeInlineState, 1000)" in html
    assert "updateRuntimeInlineStateFromStatus(payload)" in html
    assert '<button id="recover-runtime" type="button">Refresh & Reconnect</button>' in html
    assert html.index('<button id="recover-runtime"') < html.index(
        'id="runtime-inline-state"'
    )


def test_runtime_audio_monitor_component_does_not_start_runtime_twice_when_connected() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "const alreadyConnected = Boolean(" in html
    assert "const wasAudioPaused = Boolean(" in html
    assert "already connected; audio resumed" in html
    assert "already connected; waiting for runtime audio" in html
    assert "if (alreadyConnected && runtimeStartRequested)" in html
    assert "starting runtime; monitor connected" in html
    assert "runtimeStartRequested = payload.phase" in html
    assert "runtime paused; resume generation first" in html


def test_runtime_audio_monitor_component_promotes_start_button_size() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "#start-runtime" in html
    assert "width: min(100%, 430px);" in html
    assert "min-height: 40px;" in html
    assert "padding: 10px 16px;" in html
    assert "font-weight: 700;" in html


def test_runtime_audio_monitor_component_surfaces_websocket_close_and_error_status() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "connected; waiting for runtime audio" in html
    assert "closed ${event.code}${reason}" in html
    assert "websocket error; check RuntimeControlAPI and ws_url" in html
    assert "runtime start failed" in html


def test_runtime_audio_monitor_component_prevents_duplicate_active_monitors() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "BroadcastChannel" in html
    assert "runtime_audio_monitor_claim" in html
    assert "inactive; another monitor is active" in html
    assert "activeSources" in html
    assert "resetAudioPlayback" in html


def test_runtime_audio_monitor_component_restores_armed_state_after_reload() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "ARMED_STORAGE_KEY" in html
    assert "localStorage.setItem(ARMED_STORAGE_KEY, args.ws_url)" in html
    assert "localStorage.getItem(ARMED_STORAGE_KEY)" in html
    assert "isMonitorArmed()" in html
    assert "clearArmed: false" in html
    assert "pagehide" in html


def test_runtime_audio_monitor_component_reconnects_armed_websocket_without_audio_gesture() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "allowPendingAudio" in html
    assert "scheduleReconnect" in html
    assert "MAX_RECONNECT_ATTEMPTS" in html
    assert "connected; click Connect & Start audio to hear audio" in html
    assert "connectMonitor({ userGesture: false, allowPendingAudio: true })" in html
    assert "resetTimelineOnOpen: false" in html
    assert "if (resetTimelineOnOpen)" in html


def test_runtime_audio_monitor_component_waits_for_websocket_open_before_start() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "const opened = new Promise" in html
    assert "return opened" in html
    assert html.index("const connected = await connectMonitor({") < html.index(
        "window.fetch(args.start_url"
    )


def test_runtime_audio_monitor_component_normalizes_start_options_before_post() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "function normalizeStartOptions(startOptions)" in html
    assert "function parseJsonObject(value)" in html
    assert "function parseJsonArray(value)" in html
    assert "JSON.stringify(normalizeStartOptions(args.start_options || {}))" in html
    assert '"fixed_speaker_sequence" in normalized' in html
    assert "normalized.fixed_speaker_sequence = parseJsonArray(" in html
    assert "value.items || value.value || value.sequence" in html
    assert 'keys.every((key) => /^\\d+$/.test(key))' in html
    assert "trimmed.includes(\",\")" in html
    assert ".split(\",\")" in html


def test_runtime_audio_monitor_component_does_not_steal_active_audio_without_context() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "!userGesture && allowPendingAudio && !canPlayAudio()" in html
    assert "click Connect & Start audio" in html
    assert "userGesture || (isMonitorArmed() && canPlayAudio())" in html


def test_runtime_audio_monitor_component_retries_initial_streamlit_ready_notification() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "function notifyStreamlitReady()" in html
    assert "function scheduleInitialStreamlitNotifications()" in html
    assert "streamlitRenderReceived" in html
    assert "notifyUntilRender" in html
    assert "Date.now() - startedAt > 8000" in html
    assert "window.requestAnimationFrame(notifyStreamlitReady)" in html
    assert "window.setTimeout(notifyUntilRender, 250)" in html
    assert "DOMContentLoaded" in html
    assert 'window.addEventListener("load", notifyStreamlitReady' in html


def test_runtime_audio_monitor_component_handles_pcm_boundaries_without_chunk_fades() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "pcmRemainder" in html
    assert "pendingPcmChunks" in html
    assert "PLAYBACK_BATCH_TARGET_SECONDS" in html
    assert "function appendPcmChunkToPlaybackBatch(message)" in html
    assert "function flushPlaybackBatch(lane)" in html
    assert "completeByteLength" in html
    assert "item.audioChunks += playback.chunkCount || 1" in html
    assert "const playback = appendPcmChunkToPlaybackBatch(message)" in html
    assert "const playback = flushPlaybackBatch(lane)" in html
    assert "applyBoundaryFade" not in html
    assert "fadeFrames" not in html
    assert "PLAYBACK_LEAD_SECONDS" in html
    assert "context.currentTime + PLAYBACK_LEAD_SECONDS" in html


def test_runtime_audio_monitor_component_uses_interrupt_hook_for_manual_barge_in() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "runtime_interrupt_url" in html
    assert "function interruptActiveMainPlayback" in html
    assert "played_ms" in html
    assert "browser_overlap" in html
    assert "reason: \"human_barge_in\"" in html
    assert "postRuntimeControl(args.runtime_interrupt_url" in html
    assert "mainPlaybackSegments" in html
    assert "stopMainPlaybackSegments" in html
    assert "interruptedPlaybackKeys" in html
    assert "shouldDropInterruptedPlayback" in html
    assert "playbackKeyFromMetadata" in html
    assert "function shouldInterruptForHumanBargeIn(allowPlaybackBargeIn)" in html
    assert "if (!allowPlaybackBargeIn || humanInputPaused(latestRuntimeStatus))" in html
    assert "return runtimeStatusAllowsAlwaysBargeIn(latestRuntimeStatus);" in html
    assert "allowPlaybackBargeIn" in html
    assert "allowActivePlayback" in html
    assert "options.allowActivePlayback || allowPlaybackBargeIn" in html
    assert "function runtimeStatusAllowsHumanRecording(status, options = {})" in html
    assert "ignoreActivePlayback" in html
    assert "HUMAN_BARGE_IN_STATUS_WAIT_MS" in html
    assert "HUMAN_BARGE_IN_STATUS_POLL_MS" in html
    assert "transcript_interrupted" in html
    assert "function handleTranscriptInterruptedMessage(message)" in html
    assert "interrupted transcript retained" in html
    assert "item.status = \"interrupted\"" in html
    assert "transcript_invalidated" in html
    assert "function handleTranscriptInvalidatedMessage(message)" in html
    assert "function removeTimelineItemByKey(key)" in html
    assert "function stopMainPlaybackSegmentsForKey(key)" in html
    assert "invalidated stale transcript" in html
    assert "async function waitForRuntimeHumanInputAfterBargeIn()" in html
    assert "humanCaptureStreamedBufferCount" in html
    assert "function flushPendingHumanPcmCaptureToStream()" in html
    assert "flushPendingHumanPcmCaptureToStream();" in html
    assert "humanStreamingReady && humanRecording" in html
    assert "barge-in; waiting for human input slot" in html
    assert "barge-in failed; runtime not ready for human input" in html
    assert "await interruptActiveMainPlayback(" in html
    assert "await waitForRuntimeHumanInputAfterBargeIn()" in html
    assert "await startHumanRecording({ allowPlaybackBargeIn: true })" in html
    assert "async function startHumanRecording(options = {})" in html
    start_function = html.index("async function startHumanRecording(options = {})")
    start_body = html[
        start_function : html.index("function cleanupHumanRecording", start_function)
    ]
    assert "if (shouldInterruptForHumanBargeIn(allowPlaybackBargeIn))" in start_body
    assert "if (allowPlaybackBargeIn) {\n            playbackInterruptedForBargeIn" not in start_body
    assert start_body.index(
        "const pcmCaptureReady = await startHumanPcmCapture"
    ) < start_body.index("await waitForRuntimeHumanInputAfterBargeIn()")
    assert start_body.index(
        "await waitForRuntimeHumanInputAfterBargeIn()"
    ) < start_body.index("const streamOpened = await openHumanAudioStream()")


def test_runtime_audio_monitor_component_supports_human_push_to_talk_input() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "v1.2 audio monitor" in html
    assert "human_input_enabled" in html
    assert "runtime_status_url" in html
    assert "ensureRuntimeReadyForHumanRecording" in html
    assert "runtimeStatusHumanInputError" in html
    assert "runtime session expired; click Refresh & Reconnect" in html
    assert "human_audio_url" in html
    assert "human_audio_stream_url" in html
    assert "humanAudioStreamUrl" in html
    assert "openHumanAudioStream" in html
    assert "sendHumanAudioStreamChunk" in html
    assert "ensureHumanMicrophoneEnabled" in html
    assert "refreshHumanMicrophoneForRecording" in html
    refresh_start = html.index("async function refreshHumanMicrophoneForRecording()")
    refresh_end = html.index("async function canStartHumanRecording(options = {})", refresh_start)
    refresh_body = html[refresh_start:refresh_end]
    assert "humanMicrophoneIsOn()" not in refresh_body
    assert "humanInputStream.getTracks().forEach((track) => track.stop())" in refresh_body
    assert "human-talk-row" in html
    assert "human-microphone-row" in html
    assert "human-status-row" in html
    assert '<div class="meter"' not in html
    assert 'id="level"' not in html
    assert "const levelEl =" not in html
    assert "function startMeter()" not in html
    assert 'id="human-input-level"' in html
    assert 'role="meter"' in html
    assert 'aria-label="Microphone input level"' in html
    assert "human-input-level-fill" in html
    assert "function updateHumanInputLevel(peak, rms)" in html
    assert "const settings = humanAlwaysSensitivitySettings();" in html
    assert "safePeak / settings.peak" in html
    assert "safeRms / settings.rms" in html
    assert "safeRms / settings.peakRmsFloor" in html
    assert "humanCurrentPeak" in html
    assert "humanCurrentRms" in html
    assert "updateHumanInputLevel(humanCurrentPeak, humanCurrentRms);" in html
    assert "--human-input-level-width" in html
    assert "--human-input-level-color" in html
    assert "humanInputLevelEl.setAttribute(\"aria-valuenow\", String(percent));" in html
    assert "updateHumanInputLevel(streamedPeak, streamedRms);" in html
    assert "updateHumanInputLevel(0, 0);" in html
    assert html.index('<div class="human-talk-row">') < html.index(
        '<div class="human-microphone-row">'
    )
    assert html.index('id="human-input-level"') < html.index(
        '<label class="human-input-gain"'
    )
    assert html.index('<div class="human-microphone-row">') < html.index(
        '<div class="human-status-row">'
    )
    assert 'id="human-status"' not in html[
        html.index('<div class="human-talk-row">') : html.index(
            '<div class="human-microphone-row">'
        )
    ]
    assert 'id="human-microphone"' in html
    assert "Default microphone" in html
    assert "color-scheme: dark;" in html
    assert "select option" in html
    assert "background: #0b0f14;" in html
    assert "color: #f8fafc;" in html
    assert "refreshHumanMicrophones" in html
    assert "enumerateDevices" in html
    assert "deviceId = { exact: deviceId }" in html
    assert "switching microphone" in html
    assert 'id="toggle-microphone"' not in html
    assert 'id="refresh-microphones"' not in html
    assert '<button id="human-record" type="button">Hold to talk</button>' in html
    assert "navigator.mediaDevices.getUserMedia" in html
    assert "new MediaRecorder" in html
    assert "AudioWorkletNode" in html
    assert "human-pcm-capture" in html
    assert "encodeHumanFloatBuffersPcm" in html
    assert "channelCount: { ideal: 1 }" in html
    assert "echoCancellation: true" in html
    assert "noiseSuppression: true" in html
    assert "autoGainControl: false" in html
    assert "const sample = channel[sampleIndex] || 0" in html
    assert "mixed[sampleIndex] += sample" in html
    assert "channelPowers[channelIndex] += sample * sample" in html
    assert "HUMAN_CHANNEL_MIX_MIN_AVERAGE_POWER_RATIO" in html
    assert "MIN_AVERAGE_POWER_RATIO" in html
    assert "applyPreferredHumanTrackConstraints" in html
    assert "track.applyConstraints({ channelCount: 1 })" in html
    assert "mono requested; browser kept" in html
    assert "HUMAN_SILENT_MIC_AUTO_RECOVER_MS" in html
    assert "HUMAN_MIC_PROBE_MS" in html
    assert "humanMicrophoneProbeCandidates" in html
    assert "probeHumanMicrophoneSignal" in html
    assert "autoRecoverHumanMicrophoneFromSilence" in html
    assert "maybeAutoRecoverHumanMicrophoneFromSilence(elapsedMs)" in html
    assert "microphone silent; searching inputs" in html
    assert "auto microphone selected:" in html
    assert "startHumanMediaRecorderFallbackStreaming" in html
    assert "streamHumanMediaRecorderFallbackPcm" in html
    assert "media recorder fallback listening" in html
    assert "media recorder fallback peak" in html
    assert "humanMediaRecorderFallbackActive" in html
    assert "channelPeaks" in html
    assert "channelRms" in html
    assert "selectedChannelIndex" in html
    assert "scaledHumanCaptureChannels" in html
    assert "normalizeHumanCaptureMessage" in html
    assert "bestChannelPower > 0" in html
    assert "mixedPower < bestChannelPower * MIN_AVERAGE_POWER_RATIO" in html
    assert "mixed.set(input[bestChannelIndex])" in html
    assert "createScriptProcessor" not in html
    assert "async function startHumanRecording(options = {})" in html
    assert "stopHumanRecordingAndSubmit()" in html
    assert "HUMAN_MIN_RECORDING_MS" in html
    assert "humanRecordingStarting" in html
    assert "humanStopAfterStart" in html
    assert "setPointerCapture" in html
    assert "releasePointerCapture" in html
    assert "lostpointercapture" in html
    assert 'addEventListener("pointerleave"' not in html
    assert "hold longer to talk" in html
    assert "humanMicrophoneTrackDiagnostic" in html
    assert "humanCaptureDiagnostic" in html
    assert "capture frames" in html
    assert "floatSignalStats" in html
    assert "humanLevelSilenceGain" in html
    assert "humanCaptureFrameCount" in html
    assert "0.000001" in html
    assert "{ resize: false }" in html
    assert "submitHumanAudioRecording" in html
    assert "new WebSocket(streamUrl)" in html
    assert 'type: "chunk"' in html
    assert "audio_base64" in html
    assert 'recording_mode: "push_to_talk"' in html
    assert "handleHumanAudioSubmitResult(result, payload)" in html
    assert 'transcript_type: "human_final"' in html
    assert "microphone on" in html
    assert "microphone permission denied" in html
    assert "microphone captured silence" in html
    assert "resetHumanInputStreamAfterSilentCapture" in html
    assert "microphone stream reset; try again" in html
    assert "capture frames ${captured.frameCount || 0} peak ${captured.peak || 0} rms ${captured.rms || 0}" in html
    assert "selected microphone unavailable" in html
    assert "no speech recognized" in html
    assert "human_streaming_stt" in html
    assert "activeSources.size > 0" in html
    assert "AI speech is playing; wait to talk" in html


def test_runtime_audio_monitor_component_recovers_started_state_before_human_input() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "function runtimeStatusIndicatesStarted(status)" in html
    assert "function updateRuntimeStartRequestedFromStatus(status)" in html
    assert "async function recoverRuntimeStartRequestedFromStatus()" in html
    assert "const payload = await response.json();" in html
    assert "updateRuntimeStartRequestedFromStatus(payload);" in html
    assert "const runtimeStartRecovery = await recoverRuntimeStartRequestedFromStatus();" in html
    assert "if (runtimeStartRecovery.blocked)" in html
    assert "if (!runtimeStartRecovery.recovered)" in html
    assert "async function canStartHumanRecording(options = {})" in html
    assert "!(await canStartHumanRecording({" in html
    assert "allowActivePlayback," in html
    assert "runtimeStartRequested = true" in html
    assert 'phase === "running" || phase === "paused"' in html
    ensure_start = html.index("async function ensureRuntimeReadyForHumanRecording()")
    ensure_body = html[
        ensure_start : html.index("async function waitForRuntimeHumanInputAfterBargeIn()", ensure_start)
    ]
    assert "updateRuntimeStartRequestedFromStatus(status);" in ensure_body
    assert "updateRuntimeInlineStateFromStatus(status);" in ensure_body
    refresh_start = html.index("async function refreshRuntimeInlineState()")
    refresh_end = html.index("async function fetchRuntimeStatus()", refresh_start)
    refresh_body = html[refresh_start:refresh_end]
    assert "!runtimeStartRequested" not in refresh_body


def test_runtime_audio_monitor_component_keeps_started_state_on_active_stream_end() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    stream_end_start = html.index('} else if (message.type === "stream_end")')
    stream_end_end = html.index('} else if (message.type === "error")', stream_end_start)
    stream_end_body = html[stream_end_start:stream_end_end]
    assert "const status = await fetchRuntimeStatus();" in stream_end_body
    assert "updateRuntimeInlineStateFromStatus(status);" in stream_end_body
    assert "await maybeResumeAlwaysHumanRecordingFromStatus(status);" in stream_end_body
    assert 'updateRuntimeStartRequestedFromStatus({ phase: "stopped" });' in stream_end_body
    assert "runtimeStartRequested = false" not in stream_end_body
    assert "streamEnded = !runtimeStartRequested" in stream_end_body
    assert "stream end; reconnecting for next audio" in stream_end_body


def test_runtime_audio_monitor_component_supports_always_hold_to_talk_toggle() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "human-always-record" in html
    assert 'role="switch"' in html
    assert "humanAlwaysRecordingEnabled" in html
    assert "startAlwaysHumanRecording" in html
    assert "stopAlwaysHumanRecording" in html
    assert "always Hold to talk" in html
    assert 'id="human-always-sensitivity"' in html
    assert "Always sens" in html
    assert "HUMAN_ALWAYS_SENSITIVITY_STORAGE_KEY" in html
    assert "HUMAN_ALWAYS_SENSITIVITY_PROFILES" in html
    assert "setHumanAlwaysSensitivity" in html
    assert "input:checked ~ .human-always-record-track" in html
    assert "input:checked + .human-always-record-track" not in html
    assert html.index('<button id="human-record" type="button">Hold to talk</button>') < html.index(
        '<span class="human-always-record-text">Always</span>'
    )
    assert html.index('<span class="human-always-record-text">Always</span>') < html.index(
        '<span class="human-always-record-track" aria-hidden="true"></span>'
    )
    assert html.index('<span class="human-always-record-track" aria-hidden="true"></span>') < html.index(
        "Always sens"
    )
    assert "always listening" in html
    assert "if (humanAlwaysRecordingEnabled)" in html
    assert "await requestHumanRecordingStop()" in html


def test_runtime_audio_monitor_component_uses_browser_vad_for_always_recording() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "recording_mode: humanAlwaysRecordingEnabled" in html
    assert '"browser_vad"' in html
    assert "completeHumanRecordingFromServer(message)" in html
    assert "always sent; monitoring" in html
    assert "HUMAN_ALWAYS_VAD_START_PEAK" not in html
    assert "humanAlwaysVadSpeechDetected" not in html
    assert "autoStopAlwaysHumanRecording" not in html
    assert "await requestHumanRecordingStop()" in html


def test_runtime_audio_monitor_component_keeps_always_realtime_vad_armed_between_turns() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "humanAlwaysVadInputAllowed" in html
    assert "humanAlwaysVadRestartPending" in html
    assert "humanAlwaysVadSawRuntimeBusy" in html
    assert "runtimeStatusAllowsHumanRecording" in html
    assert "maybeResumeAlwaysHumanRecordingFromStatus(status)" in html
    assert "always armed; waiting for human input" in html
    assert "always armed; human input listening" in html
    assert "always starting" in html
    assert "always listening" in html
    assert "always listening; no microphone signal" in html
    assert "HUMAN_ALWAYS_NO_SIGNAL_WARNING_MS" in html
    assert "HUMAN_ALWAYS_BARGE_IN_PEAK" in html
    assert "HUMAN_ALWAYS_BARGE_IN_PEAK_RMS_FLOOR" in html
    assert "HUMAN_ALWAYS_BARGE_IN_RMS" in html
    assert "HUMAN_ALWAYS_BARGE_IN_REQUIRED_WINDOWS" in html
    assert "HUMAN_ALWAYS_BARGE_IN_MIN_ACTIVE_MS" in html
    assert "HUMAN_ALWAYS_MONITOR_PREROLL_MS" in html
    assert "HUMAN_ALWAYS_AUTO_STOP_SILENCE_MS" in html
    assert "always sent; monitoring" in html
    assert "setHumanStatus(alwaysArmedStatusForRuntimeStatus(status));" in html
    assert "alwaysArmedStatusForRuntimeStatus(latestRuntimeStatus)" in html
    assert "runtimeInlineStateEl.textContent === text" in html
    assert "if (humanStatusEl.textContent !== value)" in html
    assert "options.title !== undefined" in html
    assert "humanStatusEl.removeAttribute(\"title\")" in html
    assert "setHumanStatus(statusPrefix, {" in html
    assert "await maybeResumeAlwaysHumanRecordingFromStatus(payload)" in html
    resume_start = html.index("async function maybeResumeAlwaysHumanRecordingFromStatus")
    resume_body = html[
        resume_start : html.index("function humanMicrophoneTrackDiagnostic", resume_start)
    ]
    assert "if (!humanAlwaysVadSawRuntimeBusy)" not in resume_body
    assert "runtimeStatusAllowsHumanRecording(status, {" in resume_body
    assert "ignoreActivePlayback: true" in resume_body
    assert "humanAlwaysVadInputAllowed = true;" in resume_body
    assert 'setHumanStatus("always starting");' in resume_body
    assert "await startHumanRecording({ allowActivePlayback: true });" in resume_body
    assert resume_body.index("startHumanLevelMeter(humanInputStream);") < resume_body.index(
        "if (humanAlwaysVadRestartPending)"
    )
    assert resume_body.index("startAlwaysBargeInCaptureMonitor()") < resume_body.index(
        "if (humanAlwaysVadRestartPending)"
    )
    complete_start = html.index("function completeHumanRecordingFromServer(message)")
    complete_body = html[
        complete_start : html.index("function closeHumanAudioStream", complete_start)
    ]
    assert "const hadActiveHumanRecording = Boolean(" in complete_body
    assert "if (!hadActiveHumanRecording)" in complete_body
    assert "humanAlwaysVadRestartPending = true" in complete_body
    assert "String((message || {}).text || \"\").trim()" in complete_body
    assert "humanAlwaysVadSawRuntimeBusy = false" in html
    assert "humanAlwaysRecordingEnabled = false" in html
    assert "humanAlwaysRecordToggle.checked = false" in html


def test_runtime_audio_monitor_component_allows_always_mode_barge_in() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "function maybeStartAlwaysBargeInFromLevel(" in html
    assert "function canStartAlwaysInputFromLevel()" in html
    assert "function alwaysInputModeForCurrentRuntimeState()" in html
    assert "async function startAlwaysHumanInputFromMonitor(inputMode)" in html
    assert "function humanAlwaysBargeInSignalDetected(" in html
    assert "settings = humanAlwaysSensitivitySettings()" in html
    assert "rms >= settings.rms ||" in html
    assert "peak >= settings.peak && rms >= settings.peakRmsFloor" in html
    assert "humanAlwaysBargeInSignalElapsedMs >= settings.minActiveMs" in html
    assert "settings.requiredWindows * HUMAN_ALWAYS_BARGE_IN_SIGNAL_WINDOW_MS" in html
    assert "peak: 45" in html
    assert "rms: 6" in html
    assert "requiredWindows: 2" in html
    assert "minActiveMs: 400" in html
    assert "HUMAN_ALWAYS_FALSE_START_CANCEL_MS" in html
    assert "humanRecordingStartedByBargeIn" in html
    assert "cancelAlwaysBargeInFalseStart" in html
    assert "割り込み音声の文字起こしを確認しています。" in html
    assert "humanAlwaysBargeInControllerActive" in html
    assert "humanAlwaysPendingPreRoll" in html
    assert "mergeHumanAlwaysPreRolls(" in html
    assert "function canInterruptPlaybackFromAlwaysRecording()" in html
    assert "function maybeInterruptActivePlaybackFromAlwaysRecording(" in html
    assert "humanRecording &&" in html
    assert "maybeInterruptActivePlaybackFromAlwaysRecording(" in html
    assert 'setHumanStatus("always barge-in; interrupting playback")' in html
    assert "resetHumanAlwaysBargeInControllerState" in html
    assert "always barge-in; opening human input" in html
    assert "always speech detected; opening human input" in html
    assert "barge-in; waiting for human input slot" in html
    assert "always barge-in failed; monitoring" in html
    assert "always armed; barge-in listening" in html
    assert "capture frames ${humanCaptureFrameCount}" in html
    assert "humanCaptureDiagnostic()" in html
    assert "await interruptActiveMainPlayback(" in html
    assert '{ reason: "human_barge_in" }' in html
    assert "allowPlaybackBargeIn: true" in html
    assert "reuseMicrophone: true" not in html
    assert "function runtimeStatusAllowsAlwaysBargeIn(status)" in html
    assert "function interruptRuntimePlaybackFromStatus(status, options = {})" in html
    assert "played_ms: 0" in html
    assert "await interruptPlaybackForHumanBargeIn()" in html
    assert "alwaysInputModeForCurrentRuntimeState() ||" in html
    assert "(args.runtime_status_url ? \"status_refresh\" : \"\")" in html
    assert "always speech detected; checking runtime" in html
    assert "resolvedInputMode = alwaysInputModeForCurrentRuntimeState()" in html
    assert "allowActivePlayback: resolvedInputMode === \"barge_in\"" in html
    assert "startedByBargeIn: resolvedInputMode === \"barge_in\"" in html
    controller_start = html.index("async function startAlwaysHumanInputFromMonitor(inputMode)")
    controller_body = html[
        controller_start : html.index("async function maybeResumeAlwaysHumanRecordingFromStatus", controller_start)
    ]
    assert "let resolvedInputMode = inputMode;" in controller_body
    assert controller_body.index(
        "humanAlwaysPendingPreRoll = consumeHumanAlwaysMonitorPreRoll()"
    ) < controller_body.index("await interruptPlaybackForHumanBargeIn()")
    assert controller_body.index("await waitForRuntimeHumanInputAfterBargeIn()") < controller_body.index(
        "await startHumanRecording({"
    )
    assert "const inputGain = humanInputGainValue();" in html
    assert "const streamedPeak = Math.round(scaledPeak * inputGain);" in html
    assert "const streamedRms = Math.round(scaledRms * inputGain);" in html
    assert "const detectionPeak = Math.round(stats.peak * detectionGain);" in html
    assert "const detectionRms = Math.round(stats.rms * detectionGain);" in html
    assert "Math.round(stats.peak * 1000 * detectionGain)" not in html
    assert "Math.round(stats.rms * 1000 * detectionGain)" not in html
    assert "updateHumanInputLevel(streamedPeak, streamedRms);" in html
    assert html.count(
        "maybeStartAlwaysBargeInFromLevel(streamedPeak, streamedRms)"
    ) == 1
    assert "maybeStartAlwaysBargeInFromLevel(detectionPeak, detectionRms, durationMs)" in html
    assert "consumeHumanAlwaysMonitorPreRoll()" in html
    assert "options.alwaysPreRoll" in html
    assert "primeHumanCaptureWithPreRoll(alwaysPreRoll);" in html
    assert "appendHumanAlwaysMonitorPreRoll(buffer);" in html
    assert "startAlwaysBargeInCaptureMonitor()" in html
    assert "startHumanLevelMeter(humanInputStream);" in html

    mode_start = html.index("function alwaysInputModeForRuntimeStatus(status)")
    mode_body = html[
        mode_start : html.index("function alwaysArmedStatusForRuntimeStatus(status)", mode_start)
    ]
    assert mode_body.index("activeSources.size > 0") < mode_body.index(
        "runtimeStatusAllowsHumanRecording"
    )
    armed_start = html.index("function alwaysArmedStatusForRuntimeStatus(status)")
    armed_body = html[
        armed_start : html.index("function shouldInterruptForHumanBargeIn", armed_start)
    ]
    assert armed_body.index("activeSources.size > 0") < armed_body.index(
        "runtimeStatusAllowsHumanRecording"
    )

    stop_start = html.index("function stopMainPlaybackSegments(exceptKey)")
    stop_body = html[
        stop_start : html.index("function stopMainPlaybackSegmentsForKey(key)", stop_start)
    ]
    assert "clearAllPcmRemainders();" in stop_body


def test_runtime_audio_monitor_component_serializes_pcm_capture_startup() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "let humanPcmCaptureOpening = null;" in html
    start_index = html.index("async function startHumanPcmCapture(stream)")
    start_body = html[
        start_index : html.index("async function startHumanPcmCaptureOnce(stream)", start_index)
    ]
    assert "if (humanPcmCaptureOpening)" in start_body
    assert "return humanPcmCaptureOpening;" in start_body
    assert "const opening = startHumanPcmCaptureOnce(stream);" in start_body
    assert "humanPcmCaptureOpening = opening;" in start_body
    assert "if (humanPcmCaptureOpening === opening)" in start_body
    assert "humanPcmCaptureOpening = null;" in start_body


def test_runtime_audio_monitor_component_auto_stops_always_recording_after_silence() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "function maybeAutoStopAlwaysRecordingFromLevel(" in html
    assert "HUMAN_ALWAYS_AUTO_STOP_SPEECH_PEAK" in html
    assert "HUMAN_ALWAYS_AUTO_STOP_SPEECH_RMS" in html
    assert "HUMAN_ALWAYS_AUTO_STOP_SPEECH_CONFIRM_MS" in html
    assert "HUMAN_ALWAYS_AUTO_STOP_MIN_RECORDING_MS" in html
    assert "HUMAN_ALWAYS_AUTO_STOP_SILENCE_PEAK" in html
    assert "HUMAN_ALWAYS_AUTO_STOP_SILENCE_RMS" in html
    assert "always detected silence; sending" in html
    assert "void stopHumanRecordingAndSubmit();" in html
    assert "maybeAutoStopAlwaysRecordingFromLevel(" in html
    assert "elapsedMs >= HUMAN_ALWAYS_AUTO_STOP_MIN_RECORDING_MS" in html
    assert "HUMAN_ALWAYS_AUTO_STOP_MAX_SPEECH_MS" not in html
    assert "humanAlwaysAutoStopSpeechActiveMs += durationMs;" in html
    assert "const detectionPeak = Math.round(stats.peak * detectionGain);" in html
    assert "const detectionRms = Math.round(stats.rms * detectionGain);" in html
    assert "always sent; monitoring" in html


def test_runtime_audio_monitor_component_avoids_empty_stream_commit_on_always_stop() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "const HUMAN_AUDIO_STREAM_MIN_END_MS = 120;" in html
    assert "function sendHumanAudioStreamSilenceChunk(socket)" in html
    assert "const HUMAN_AUDIO_STREAM_END_ACK_WAIT_MS = 900;" in html
    assert "function noteHumanAudioStreamChunkAck(chunkIndex)" in html
    assert "function waitForHumanAudioStreamChunkAck(chunkIndex)" in html
    assert 'if (message.type === "chunk_received")' in html
    assert "noteHumanAudioStreamChunkAck(message.chunk_index);" in html

    close_start = html.index("function closeHumanAudioStream")
    close_body = html[
        close_start : html.index(
            "function stopHumanMediaRecorderFallbackStreaming", close_start
        )
    ]
    assert "if (humanStreamingChunkIndex <= 0)" in close_body
    assert close_body.index("sendHumanAudioStreamSilenceChunk(socket);") < close_body.index(
        "await waitForHumanAudioStreamChunkAck(finalChunkIndex);"
    )
    assert "const finalChunkAcknowledged =" in close_body
    assert "!finalChunkAcknowledged &&" in close_body
    assert close_body.index("await waitForHumanAudioStreamChunkAck(finalChunkIndex);") < close_body.index(
        'socket.send(JSON.stringify({ type: "end" }));'
    )
    assert "window.setTimeout(() =>" in close_body

    send_start = html.index("function sendHumanAudioStreamChunk(")
    send_body = html[
        send_start : html.index("function sendHumanAudioStreamSilenceChunk", send_start)
    ]
    assert '{ requireRecording = true, source = "pcm" } = {}' in send_body
    assert "(requireRecording && !humanRecording)" in send_body

    flush_start = html.index("function flushCapturedHumanPcmCaptureToStream(captured)")
    flush_body = html[
        flush_start : html.index("function resumeAlwaysMonitorIfIdle", flush_start)
    ]
    assert "!captured.sampleRate" in flush_body
    assert "!humanCaptureSampleRate" not in flush_body
    assert "{ requireRecording: false }" in flush_body

    stop_start = html.index("async function stopHumanRecordingAndSubmit()")
    streaming_branch = html[
        stop_start : html.index("if (!recorder)", stop_start)
    ]
    assert streaming_branch.index("const captured = stopHumanPcmCapture();") < streaming_branch.index(
        "flushCapturedHumanPcmCaptureToStream(captured);"
    )
    assert streaming_branch.index(
        "flushCapturedHumanPcmCaptureToStream(captured);"
    ) < streaming_branch.index("await closeHumanAudioStream({ sendEnd: true });")

    capture_stop_start = html.index("function stopHumanPcmCapture()")
    capture_stop_body = html[
        capture_stop_start : html.index("function preferredHumanRecorderMimeType", capture_stop_start)
    ]
    assert "const streamedBufferCount = humanCaptureStreamedBufferCount || 0;" in capture_stop_body
    assert (
        "return { buffers, sampleRate, frameCount, peak, rms, streamedBufferCount };"
        in capture_stop_body
    )


def test_runtime_audio_monitor_component_completes_recording_on_its_input_socket() -> (
    None
):
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    transcript_start = html.index("function handleTranscriptMessage(message, final)")
    transcript_body = html[
        transcript_start : html.index(
            "function handleTranscriptInterruptedMessage", transcript_start
        )
    ]
    assert "completeHumanRecordingFromServer(message);" not in transcript_body
    stream_start = html.index("function openHumanAudioStream()")
    stream_body = html[
        stream_start : html.index(
            "function maybeAutoStopAlwaysRecordingFromLevel", stream_start
        )
    ]
    assert "if (humanStreamingSocket !== socket)" in stream_body
    assert 'if (message.type === "completed")' in stream_body
    assert "completeHumanRecordingFromServer(message);" in stream_body


def test_runtime_audio_monitor_component_queues_transcript_on_audio_context_timeline() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "Live conversation timeline" in html
    assert "const timelineItems = new Map()" in html
    assert "max-height: 280px;" in html
    assert "font-size: 24px;" in html
    assert "function keepTimelineScrolledToBottom()" in html
    assert "timelineLanesEl.scrollTop = timelineLanesEl.scrollHeight" in html
    assert "function resetTimelineDisplay()" in html
    assert html.count("timelineItems.clear()") == 1
    assert "function renderTimeline(options)" in html
    assert "forceScrollToBottom: false" in html
    assert "function handleTranscriptMessage(message, final)" in html
    assert "renderTimeline({ forceScrollToBottom: item.visibleWithoutPlayback })" in html
    assert "function updateTimelineAudio(metadata, playback)" in html
    assert "function isHumanTranscriptMetadata(metadata)" in html
    assert "item.visibleWithoutPlayback" in html
    assert "context.currentTime + PLAYBACK_LEAD_SECONDS >= item.startAt" in html
    assert 'message.type === "transcript_delta"' in html
    assert 'message.type === "transcript_final"' in html
    assert "timeline-lane" in html


def test_runtime_audio_monitor_component_sorts_timeline_by_logical_turn_order() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "function timelineTurnSortValue(item)" in html
    assert "function compareTimelineItems(left, right)" in html
    assert "return leftTurn - rightTurn;" in html
    assert (
        "return leftStart - rightStart || left.createdOrder - right.createdOrder;"
        in html
    )
    assert ".sort(compareTimelineItems)" in html


def test_runtime_audio_monitor_component_allows_manual_timeline_scroll_when_not_playing() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "function timelineShouldAutoScroll()" in html
    assert "activeSources.size > 0" in html
    assert "!audioPlaybackPaused" in html
    assert "function stopTimelineTimer()" in html
    assert "function refreshTimelineTimer()" in html
    assert "restoreTimelineScrollPosition(previousScrollTop)" in html
    assert "if (timelineShouldAutoScroll())" in html
    assert "refreshTimelineTimer();" in html


def test_runtime_audio_monitor_component_emits_playback_completion_events() -> None:
    html = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "streamlit:setComponentValue" not in html
    assert "function setComponentValue(value)" not in html
    assert "runtime_playback_completed_url" in html
    assert "postRuntimeControl(" in html
    assert "args.runtime_playback_completed_url" in html
    assert "runtime_playback_ended" in html
    assert "emitPlaybackCompletedIfReady" in html
    assert "turnEndPlaybackKeys" in html
    assert "completedPlaybackKeys" in html
    assert "playbackMetadataByKey" in html
    assert "runtime_audio_paused" not in html
    assert "runtime_audio_resumed" not in html
    assert "runtime_audio_started" not in html
    assert "runtime_audio_reconnected" not in html
    assert 'event_type: "playback_ended"' not in html
    assert "playback_skipped" not in html
