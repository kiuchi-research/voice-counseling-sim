const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const html = fs.readFileSync(
  path.join(__dirname, "../app/components/runtime_audio_monitor/index.html"),
  "utf8",
);

function loadFunctions(context, first, next) {
  const start = html.indexOf(first);
  const end = html.indexOf(next, start);
  assert.ok(start >= 0 && end > start, first);
  vm.runInNewContext(html.slice(start, end), context);
}

function inputContext() {
  const sockets = [];
  class Socket {
    static OPEN = 1;
    static CONNECTING = 0;
    static CLOSING = 2;
    static CLOSED = 3;
    constructor() {
      this.readyState = Socket.CONNECTING;
      this.listeners = new Map();
      this.sent = [];
      sockets.push(this);
    }
    addEventListener(type, listener) {
      if (!this.listeners.has(type)) this.listeners.set(type, []);
      this.listeners.get(type).push(listener);
    }
    send(payload) { this.sent.push(JSON.parse(payload)); }
    close() { this.readyState = Socket.CLOSING; }
    emit(type, payload = {}) {
      if (type === "open") this.readyState = Socket.OPEN;
      if (type === "close") this.readyState = Socket.CLOSED;
      for (const listener of this.listeners.get(type) || []) {
        listener(type === "message" ? { data: JSON.stringify(payload) } : payload);
      }
    }
  }
  let nextId = 0;
  const context = {
    WebSocket: Socket,
    window: { WebSocket: Socket, setTimeout: () => 1, clearTimeout() {} },
    performance: { now: () => 10000 },
    humanRecording: false,
    humanRecordingStarting: false,
    humanRecordingAttemptId: 0,
    humanInputPaused: () => false,
    humanRecordingStopping: false,
    humanRecordingStartedAt: 1000,
    humanRecordingStartedByBargeIn: false,
    humanStopAfterStart: false,
    humanAlwaysRecordingEnabled: true,
    humanAlwaysAutoStopAudioMs: 0,
    humanAlwaysAiResumePending: false,
    humanAlwaysVadInputAllowed: true,
    humanAlwaysVadRestartPending: false,
    humanAlwaysVadSawRuntimeBusy: false,
    humanMediaRecorder: null,
    humanRecordingChunks: [],
    humanStreamingSocket: null,
    humanStreamingReady: false,
    humanStreamingEnded: true,
    humanStreamingClientMessageId: "",
    humanStreamingError: "",
    humanMediaRecorderFallbackActive: false,
    humanMediaRecorderFallbackDecoding: null,
    humanMediaRecorderFallbackPcmBytesSent: 0,
    latestRuntimeStatus: {},
    activeSources: new Set(),
    runtimeStatusHumanInputError: () => "",
    runtimeStatusAllowsAlwaysBargeIn: () => false,
    runtimeStatusAllowsHumanRecording: () => true,
    humanStreamingChunkIndex: 0,
    humanStreamingChunkAckIndex: -1,
    humanStreamingChunkAckWaiters: [],
    HUMAN_MIN_RECORDING_MS: 200,
    humanRecordButton: { classList: { add() {}, remove() {} } },
    humanInputStream: { getAudioTracks: () => [{}] },
    audioContext: null,
    humanAudioStreamUrl: () => "ws://local.test/runtime/human-audio-stream",
    humanSessionId: () => "session",
    humanClientMessageId: () => `stream-${++nextId}`,
    humanRecipientIds: () => ["client_a"],
    humanSpeakerId: () => "counselor",
    humanSpeakerDisplayName: () => "カウンセラー",
    runtimeIsHumanClientMode: () => false,
    bytesToBase64: (bytes) => Buffer.from(bytes).toString("base64"),
    waitForHumanAudioStreamChunkAck: async () => true,
    canStartHumanRecording: async () => true,
    ensureRuntimeReadyForHumanRecording: async () => true,
    shouldInterruptForHumanBargeIn: () => false,
    ensureHumanMicrophoneEnabled: async () => true,
    refreshHumanMicrophoneForRecording: async () => true,
    startHumanPcmCapture: async () => true,
    stopHumanPcmCapture: () => ({ frameCount: 2000, peak: 40 }),
    mergeHumanAlwaysPreRolls: () => null,
    consumeHumanAlwaysMonitorPreRoll: () => null,
    timelineItemForMetadata: () => ({ visibleWithoutPlayback: true, startAt: null }),
    isHumanTranscriptMetadata: () => true,
    statuses: [],
    setHumanStatus(message) { context.statuses.push(message); },
    describeMicrophoneError: (error) => error.message,
  };
  for (const name of [
    "clearHumanAudioStreamChunkAckWaiters", "noteHumanAudioStreamChunkAck",
    "updateHumanRecordButtonLabel", "stopHumanMediaRecorderFallbackStreaming",
    "stopHumanLevelMeter", "resetHumanAlwaysAutoStopDetection",
    "resetHumanAlwaysBargeInControllerState", "resetHumanAlwaysPendingPreRoll",
    "resetHumanAlwaysRestartState", "resumeAlwaysMonitorIfIdle",
    "resetHumanAlwaysBargeInDetection", "primeHumanCaptureWithPreRoll",
    "startHumanLevelMeter", "flushPendingHumanPcmCaptureToStream",
    "flushCapturedHumanPcmCaptureToStream",
    "setRuntimeInlineState", "renderTimeline",
  ]) context[name] = () => {};
  loadFunctions(context, "function openHumanAudioStream()", "function maybeAutoStopAlwaysRecordingFromLevel(");
  loadFunctions(context, "function completeHumanRecordingFromServer(message)", "function noteHumanAudioStreamChunkAck(");
  loadFunctions(context, "async function closeHumanAudioStream(", "function stopHumanMediaRecorderFallbackStreaming()");
  loadFunctions(context, "function sendHumanAudioStreamChunk(", "function sendHumanAudioStreamSilenceChunk(");
  loadFunctions(context, "async function startHumanRecording(", "async function stopHumanRecordingAndSubmit()");
  loadFunctions(context, "async function stopHumanRecordingAndSubmit()", "function startHumanLevelMeter(stream)");
  loadFunctions(context, "async function requestHumanRecordingStop()", "async function startAlwaysHumanRecording()");
  loadFunctions(context, "function handleTranscriptMessage(message, final)", "function handleTranscriptInterruptedMessage(");
  loadFunctions(context, "function alwaysArmedStatusForRuntimeStatus(status)", "function shouldInterruptForHumanBargeIn(");
  return { context, sockets };
}

async function connect(input, turnId) {
  const { context, sockets } = input;
  context.humanRecording = true;
  const opening = context.openHumanAudioStream();
  const socket = sockets.at(-1);
  socket.emit("open");
  socket.emit("message", {
    type: "accepted", turn_id: turnId, stream_id: context.humanStreamingClientMessageId,
  });
  assert.equal(await opening, true);
  return socket;
}

for (const delayedEvent of ["close", "completed"]) {
  test(`late ${delayedEvent} from a finished turn cannot stop the next microphone stream`, async () => {
    const input = inputContext();
    const { context } = input;
    const old = await connect(input, 1);
    old.emit("message", { type: "completed", stream_id: "stream-1", turn_id: 1 });
    const current = await connect(input, 3);
    if (delayedEvent === "close") old.emit("close");
    else old.emit("message", { type: "completed", stream_id: "stream-1", turn_id: 1 });
    assert.equal(context.humanRecording, true);
    assert.equal(context.humanStreamingReady, true);
    assert.equal(context.humanStreamingSocket, current);
    assert.equal(context.sendHumanAudioStreamChunk(new Uint8Array([1, 0])), 0);
    assert.equal(current.sent.at(-1).type, "chunk");
  });
}

for (const eventType of ["error", "close"]) {
  test(`a ${eventType} after acceptance clears recording and reports the failure`, async () => {
    const input = inputContext();
    const { context } = input;
    const socket = await connect(input, 1);
    if (eventType === "error") socket.emit("message", { type: "error", message: "stream rejected" });
    else socket.emit("close");
    assert.equal(context.humanRecording, false);
    assert.equal(context.humanStreamingSocket, null);
    assert.equal(context.humanStreamingReady, false);
    assert.match(context.statuses.at(-1), /rejected|closed/);
    const next = await connect(input, 1);
    assert.equal(context.humanStreamingSocket, next);
    assert.equal(context.sendHumanAudioStreamChunk(new Uint8Array([1, 0])), 0);
  });
}

test("monitor transcripts cannot finish a recording; its own socket completion does", async () => {
  const input = inputContext();
  const { context } = input;
  const socket = await connect(input, 3);
  context.handleTranscriptMessage({ metadata: { session_id: "session", turn_id: 1 }, text: "previous" }, true);
  assert.equal(context.humanRecording, true);
  assert.equal(context.humanStreamingSocket, socket);
  context.handleTranscriptMessage({ metadata: { session_id: "session", turn_id: 3 }, text: "current" }, true);
  assert.equal(context.humanRecording, true);
  socket.emit("message", { type: "completed", stream_id: "stream-1", turn_id: 3 });
  assert.equal(context.humanRecording, false);
});

test("recording startup is reserved before its first asynchronous check", async () => {
  const { context } = inputContext();
  let checks = 0;
  let release;
  const ready = new Promise((resolve) => { release = resolve; });
  context.canStartHumanRecording = () => { checks += 1; return ready; };
  const first = context.startHumanRecording();
  const second = context.startHumanRecording();
  release(false);
  await Promise.all([first, second]);
  assert.equal(checks, 1);
  assert.equal(context.humanRecordingStarting, false);
});

test("releasing the talk button while the microphone is opening waits for startup", async () => {
  const { context } = inputContext();
  context.humanRecordingStarting = true;
  context.humanRecording = true;
  let stopped = false;
  context.stopHumanRecordingAndSubmit = async () => { stopped = true; };
  await context.requestHumanRecordingStop();
  assert.equal(stopped, false);
  assert.equal(context.humanStopAfterStart, true);
});

test("an old end request waiting for an acknowledgement cannot mutate a new stream", async () => {
  const input = inputContext();
  const { context } = input;
  const old = await connect(input, 1);
  context.humanStreamingChunkIndex = 1;
  let release;
  context.waitForHumanAudioStreamChunkAck = () => new Promise((resolve) => { release = resolve; });
  const ending = context.closeHumanAudioStream({ sendEnd: true });
  context.completeHumanRecordingFromServer({ turn_id: 1, stream_id: "stream-1" });
  const current = await connect(input, 3);
  release(true);
  await ending;
  assert.equal(context.humanStreamingEnded, false);
  assert.equal(context.humanStreamingReady, true);
  assert.equal(context.humanStreamingSocket, current);
  assert.equal(old.sent.some((event) => event.type === "end"), false);
});

test("a failed submission stays visible and releases the recording stop lock", async () => {
  const input = inputContext();
  const { context } = input;
  context.humanAlwaysRecordingEnabled = false;
  const socket = await connect(input, 1);
  context.humanStreamingChunkIndex = 1;
  let release;
  context.waitForHumanAudioStreamChunkAck = () => new Promise((resolve) => { release = resolve; });
  const stopping = context.stopHumanRecordingAndSubmit();
  socket.emit("message", { type: "error", message: "stream rejected" });
  release(false);
  await stopping;
  assert.equal(context.humanRecordingStopping, false);
  assert.equal(context.humanRecording, false);
  assert.equal(context.statuses.at(-1), "stream rejected");
});

for (const always of [true, false]) {
  test(`${always ? "Always" : "Hold to talk"} can start, send, finish, and start the next turn`, async () => {
    const { context, sockets } = inputContext();
    context.humanAlwaysRecordingEnabled = always;
    for (const turnId of [1, 3]) {
      const previous = sockets.at(-1);
      const socketCount = sockets.length;
      const starting = context.startHumanRecording();
      for (let i = 0; i < 30 && sockets.length === socketCount; i += 1) {
        await Promise.resolve();
      }
      assert.equal(sockets.length, socketCount + 1);
      const socket = sockets.at(-1);
      socket.emit("open");
      socket.emit("message", { type: "accepted", turn_id: turnId });
      await starting;
      if (previous) {
        previous.emit("message", { type: "completed", turn_id: 1 });
        previous.emit("close");
      }
      assert.equal(context.humanRecording, true);
      assert.equal(context.sendHumanAudioStreamChunk(new Uint8Array([1, 0])), 0);
      assert.equal(socket.sent.at(-1).type, "chunk");
      if (!always) {
        await context.requestHumanRecordingStop();
        assert.equal(socket.sent.at(-1).type, "end");
      }
      socket.emit("message", { type: "completed", turn_id: turnId });
      assert.equal(context.humanRecording, false);
      assert.equal(context.humanRecordingStarting, false);
      assert.equal(context.humanRecordingStopping, false);
      assert.equal(context.humanStreamingSocket, null);
    }
  });
}

function roleContext(mode = "ai_counselor_human_client") {
  const human = mode === "ai_counselor_human_client" ? "client" : "counselor";
  const recipient = human === "client" ? "counselor" : "client";
  const context = {
    args: { human_input_enabled: true, start_options: { interaction_mode: mode } },
    latestRuntimeStatus: {
      phase: "running", interaction_mode: mode, session_id: "live",
      human_speaker_id: human, human_recipient_ids: [recipient],
      participants: {
        [human]: { role: human, actor_kind: "human", display_name: "参加者" },
        [recipient]: { role: recipient, actor_kind: "ai", display_name: "AI" },
      },
      current_speaker: human, awaiting_human_input: true,
    },
    activeSources: new Set(), warmupActive: false, audioPlaybackPaused: false,
    humanRecording: false, humanRecordingStarting: false, humanAlwaysRecordingEnabled: true,
  };
  loadFunctions(context, "function runtimeInteractionMode(", "function updateRuntimeInlineStateFromStatus(");
  loadFunctions(context, "function humanSpeakerDisplayName(", "function scheduleReconnect()");
  return context;
}

for (const mode of ["ai_counselor_human_client", "human_counselor_ai_client"]) {
  test(`${mode}: human role determines recording and interruption targets`, () => {
    const c = roleContext(mode);
    const status = c.latestRuntimeStatus;
    assert.equal(c.runtimeStatusAllowsHumanRecording(status), true);
    assert.equal(c.runtimeStatusAllowsAlwaysBargeIn(status), false);
    assert.equal(c.humanSpeakerId(status), status.human_speaker_id);
    assert.equal(c.humanSpeakerDisplayName(status.human_speaker_id), "参加者");
    assert.equal(c.humanRecipientIds().join(), status.human_recipient_ids.join());
    status.current_speaker = status.human_recipient_ids[0];
    status.awaiting_human_input = false;
    assert.equal(c.runtimeStatusAllowsHumanRecording(status), false);
    assert.equal(c.runtimeStatusAllowsAlwaysBargeIn(status), true);
  });
}

test("running session identity overrides changed next-session options", () => {
  const c = roleContext();
  c.args.start_options = { interaction_mode: "human_counselor_ai_client", participants: { client_a: { actor_kind: "ai" } } };
  c.args.human_input_enabled = false;
  assert.equal(c.humanSpeakerId(), "client");
  assert.equal(c.humanRecipientIds().join(), "counselor");
  assert.equal(c.runtimeStatusAllowsHumanRecording(c.latestRuntimeStatus), true);
});

test("paused human-client session cannot record or trigger Always interruption", () => {
  const c = roleContext();
  c.latestRuntimeStatus.phase = "paused";
  c.activeSources.add("paused-audio");
  assert.equal(c.runtimeStatusAllowsHumanRecording(c.latestRuntimeStatus), false);
  assert.equal(c.runtimeStatusAllowsAlwaysBargeIn(c.latestRuntimeStatus), false);
  assert.equal(c.alwaysInputModeForRuntimeStatus(c.latestRuntimeStatus), "");
  assert.equal(c.shouldInterruptForHumanBargeIn(true), false);
});

test("canceling a recording sends abort, including after end was already sent", async () => {
  for (const ended of [false, true]) {
    const input = inputContext();
    const { context } = input;
    context.runtimeIsHumanClientMode = () => true;
    const socket = await connect(input, 2);
    context.humanStreamingEnded = ended;
    await context.closeHumanAudioStream({ sendEnd: false });
    assert.equal(socket.sent.at(-1).type, "abort");
    assert.equal(socket.sent.filter(message => message.type === "end").length, 0);
    assert.equal(context.humanStreamingSocket, null);
  }
});

test("Pause while microphone permission is pending cancels startup", async () => {
  const { context: c, sockets } = inputContext();
  let resolveMic;
  c.ensureHumanMicrophoneEnabled = () => new Promise(resolve => { resolveMic = resolve; });
  const starting = c.startHumanRecording();
  while (!resolveMic) await Promise.resolve();
  c.cleanupHumanRecording();
  resolveMic(true);
  await starting;
  assert.equal(c.humanRecording, false);
  assert.equal(c.humanRecordingStarting, false);
  assert.equal(sockets.length, 0);
});

test("blank or aborted completion is not displayed as sent", async () => {
  for (const reason of ["blank_transcript", "aborted"]) {
    const input = inputContext();
    const socket = await connect(input, 2);
    socket.emit("message", { type: "completed", completion_reason: reason, text: "" });
    assert.doesNotMatch(input.context.statuses.at(-1), /sent/);
    assert.equal(input.context.humanRecording, false);
  }
});

test("human client waits for transcription after end, including the Always meter status", async () => {
  const input = inputContext();
  const { context: c } = input;
  c.runtimeIsHumanClientMode = () => true;
  const socket = await connect(input, 2);
  c.sendHumanAudioStreamChunk(new Uint8Array([1, 0]));
  await c.stopHumanRecordingAndSubmit();
  assert.equal(socket.sent.at(-1).type, "end");
  assert.match(c.statuses.at(-1), /文字起こし/);
  socket.emit("message", { type: "end_received", pending_transcription: true });
  assert.match(c.statuses.at(-1), /文字起こし/);
  assert.match(c.alwaysArmedStatusForRuntimeStatus({}), /文字起こし/);
  socket.emit("message", { type: "completed", completion_reason: "transcribed", text: "相談です" });
  assert.match(c.statuses.at(-1), /送信/);
  assert.doesNotMatch(c.alwaysArmedStatusForRuntimeStatus({}), /文字起こし/);
});

test("blank completion during end acknowledgement remains visible until the next input", async () => {
  const input = inputContext();
  const { context: c } = input;
  c.runtimeIsHumanClientMode = () => true;
  const socket = await connect(input, 2);
  c.sendHumanAudioStreamChunk(new Uint8Array([1, 0]));
  c.waitForHumanAudioStreamChunkAck = async () => {
    socket.emit("message", { type: "completed", completion_reason: "blank_transcript" });
    return true;
  };
  await c.stopHumanRecordingAndSubmit();
  assert.match(c.statuses.at(-1), /認識できません/);
  assert.match(c.alwaysArmedStatusForRuntimeStatus({}), /認識できません/);
  await connect(input, 2);
  assert.doesNotMatch(c.alwaysArmedStatusForRuntimeStatus({}), /認識できません/);
});

function recorderContext(input) {
  const c = input.context;
  c.humanCapturePeak = 0;
  c.humanCaptureRms = 0;
  c.humanMediaRecorder = { mimeType: "audio/webm", state: "recording" };
  c.humanRecordingChunks = [{}];
  c.decodeHumanRecordingToPcm = async () => new Uint8Array([7, 0, 9, 0]);
  c.pcmSignalStats = () => ({ peak: 10, rms: 7 });
  c.pcmHasSignal = () => true;
  loadFunctions(c, "async function streamHumanMediaRecorderFallbackPcm()", "function startHumanMediaRecorderFallbackStreaming()");
  return c;
}

test("silent PCM input does not let the recorder send the same interval a second time", async () => {
  const input = inputContext();
  const socket = await connect(input, 2);
  const c = recorderContext(input);
  c.sendHumanAudioStreamChunk(new Uint8Array([0, 0]));
  await c.streamHumanMediaRecorderFallbackPcm();
  assert.equal(socket.sent.filter(e => e.type === "chunk").length, 1);
  assert.equal(c.humanMediaRecorderFallbackActive, false);
});

test("PCM arriving while the recorder is decoding retains exclusive ownership", async () => {
  const input = inputContext();
  const socket = await connect(input, 2);
  const c = recorderContext(input);
  let release;
  c.decodeHumanRecordingToPcm = () => new Promise(resolve => { release = resolve; });
  const pending = c.streamHumanMediaRecorderFallbackPcm();
  c.sendHumanAudioStreamChunk(new Uint8Array([0, 0]));
  release(new Uint8Array([7, 0]));
  await pending;
  assert.equal(socket.sent.filter(e => e.type === "chunk").length, 1);
  assert.equal(c.humanMediaRecorderFallbackActive, false);
});

test("a selected recorder source excludes PCM and serializes decoding", async () => {
  const input = inputContext();
  const socket = await connect(input, 2);
  const c = recorderContext(input);
  let release;
  let decodes = 0;
  c.decodeHumanRecordingToPcm = () => {
    decodes += 1;
    return new Promise(resolve => { release = resolve; });
  };
  const first = c.streamHumanMediaRecorderFallbackPcm();
  const second = c.streamHumanMediaRecorderFallbackPcm();
  assert.equal(decodes, 1);
  release(new Uint8Array([7, 0]));
  await Promise.all([first, second]);
  assert.equal(c.sendHumanAudioStreamChunk(new Uint8Array([8, 0])), -1);
  assert.equal(socket.sent.filter(e => e.type === "chunk").length, 1);
});

test("a decode finishing after cancellation cannot inject old audio into a new stream", async () => {
  const input = inputContext();
  const old = await connect(input, 2);
  const c = recorderContext(input);
  let release;
  c.decodeHumanRecordingToPcm = () => new Promise(resolve => { release = resolve; });
  const pending = c.streamHumanMediaRecorderFallbackPcm();
  old.emit("message", { type: "completed", completion_reason: "aborted" });
  const current = await connect(input, 2);
  recorderContext(input);
  release(new Uint8Array([7, 0]));
  await pending;
  assert.equal(current.sent.filter(e => e.type === "chunk").length, 0);
});

test("interrupt acknowledgements preserve the session and current microphone stream", async () => {
  const input = inputContext();
  const { context: c } = input;
  c.runtimeIsHumanClientMode = () => true;
  c.latestRuntimeStatus = { session_id: "session", phase: "running", current_turn_id: 14 };
  c.updateHumanInputVisibility = () => {};
  c.runtimeInlineStateForStatus = () => ({ message: "", variant: "" });
  loadFunctions(c, "function updateRuntimeInlineStateFromStatus(status)", "async function refreshRuntimeInlineState()");
  const socket = await connect(input, 14);
  c.updateRuntimeInlineStateFromStatus({
    cancel_sent: true, truncate_sent: true,
    human_audio_stream_active: true, human_turn_id: 14,
  });
  assert.equal(c.humanRecording, true);
  assert.equal(c.humanStreamingSocket, socket);
  assert.equal(c.latestRuntimeStatus.session_id, "session");
  assert.equal(socket.sent.some(e => e.type === "abort"), false);
  c.updateRuntimeInlineStateFromStatus({ session_id: "next-session", phase: "running" });
  assert.equal(c.humanRecording, false);
  assert.equal(socket.sent.at(-1).type, "abort");
});

test("Always sends microphone audio throughout an HTTP playback interruption", async () => {
  const input = inputContext();
  const { context: c } = input;
  c.runtimeIsHumanClientMode = () => true;
  c.latestRuntimeStatus = { session_id: "session", phase: "running", current_turn_id: 14 };
  c.updateHumanInputVisibility = () => {};
  c.runtimeInlineStateForStatus = () => ({ message: "", variant: "" });
  c.updateRuntimeStartRequestedFromStatus = () => {};
  c.setStatus = () => {};
  loadFunctions(c, "function updateRuntimeInlineStateFromStatus(status)", "async function refreshRuntimeInlineState()");
  loadFunctions(c, "async function postRuntimeControl(", "async function pauseRuntimeGenerationForPlayback()");
  const socket = await connect(input, 14);
  let respond;
  c.window.fetch = () => new Promise(resolve => { respond = resolve; });
  const interrupt = c.postRuntimeControl("/runtime/interrupt", "failed", {
    speaker: "counselor", turn_id: 13, played_ms: 1000, reason: "human_barge_in",
  });
  c.sendHumanAudioStreamChunk(new Uint8Array([1, 0]));
  respond({ ok: true, text: async () => JSON.stringify({
    human_audio_stream_active: true, human_turn_id: 14, cancel_sent: true,
  }) });
  assert.equal(await interrupt, true);
  assert.equal(c.sendHumanAudioStreamChunk(new Uint8Array([2, 0])), 1);
  assert.equal(socket.sent.some(e => e.type === "abort"), false);
  assert.equal(c.latestRuntimeStatus.session_id, "session");
});

test("a stopped recorder cannot append its delayed final blob to the next recording", async () => {
  const input = inputContext();
  const { context: c, sockets } = input;
  const recorders = [];
  class Recorder {
    constructor() { this.listeners = {}; this.state = "inactive"; recorders.push(this); }
    addEventListener(type, handler) { this.listeners[type] = handler; }
    start() { this.state = "recording"; }
    stop() { this.state = "inactive"; }
    emit(data) { this.listeners.dataavailable({ data }); }
  }
  c.MediaRecorder = c.window.MediaRecorder = Recorder;
  c.preferredHumanRecorderMimeType = () => "audio/webm";
  c.startHumanMediaRecorderFallbackStreaming = () => {};
  for (let turn = 0; turn < 2; turn += 1) {
    const count = sockets.length;
    const starting = c.startHumanRecording();
    for (let i = 0; i < 30 && sockets.length === count; i += 1) await Promise.resolve();
    const socket = sockets.at(-1);
    socket.emit("open");
    socket.emit("message", { type: "accepted", turn_id: turn * 2 + 2 });
    await starting;
    if (turn === 0) socket.emit("message", { type: "completed", completion_reason: "transcribed" });
  }
  recorders[0].emit({ size: 20, id: "old" });
  recorders[1].emit({ size: 20, id: "current" });
  assert.deepEqual(Array.from(c.humanRecordingChunks, data => data.id), ["current"]);
});

function alwaysStopContext() {
  let now = 10000;
  const context = {
    Date: { now: () => now },
    humanAlwaysRecordingEnabled: true,
    humanRecording: true,
    humanStreamingReady: true,
    humanAlwaysAutoStopRequested: false,
    humanAlwaysAutoStopSpeechActiveMs: 0,
    humanAlwaysAutoStopLastSampleMs: 0,
    humanAlwaysAutoStopAudioMs: 0,
    humanAlwaysAutoStopSpeechDetected: false,
    humanAlwaysAutoStopLastSpeechAt: 0,
    humanRecordingStartedByBargeIn: false,
    submissions: 0,
    falseStarts: 0,
    setHumanStatus() {},
    stopHumanRecordingAndSubmit() { context.submissions += 1; },
    cancelAlwaysBargeInFalseStart() { context.falseStarts += 1; },
  };
  for (const [, name, value] of html.matchAll(/const (HUMAN_ALWAYS_(?:AUTO_STOP|FALSE_START)_[A-Z_]+) = (\d+);/g)) {
    context[name] = Number(value);
  }
  loadFunctions(context, "function maybeAutoStopAlwaysRecordingFromLevel(", "function completeHumanRecordingFromServer(");
  return {
    context,
    sample(elapsed, peak = 100, rms = 20, durationMs = 100) {
      now = 10000 + elapsed;
      return context.maybeAutoStopAlwaysRecordingFromLevel(peak, rms, elapsed, durationMs);
    },
  };
}

test("Always keeps continuous speech past nine seconds and sends once after silence", () => {
  const { context: c, sample } = alwaysStopContext();
  for (let elapsed = 1000; elapsed <= 30000; elapsed += 100) {
    assert.equal(sample(elapsed), false, `speech cut off at ${elapsed} ms`);
  }
  assert.equal(c.submissions, 0);
  for (let elapsed = 30100; elapsed < 31400; elapsed += 100) {
    assert.equal(sample(elapsed, 0, 0), false);
  }
  assert.equal(sample(31400, 0, 0), true);
  assert.equal(c.submissions, 1);
  sample(32000, 0, 0);
  assert.equal(c.submissions, 1);
});

test("Always allows a one-second pause and continuing speech", () => {
  const { context: c, sample } = alwaysStopContext();
  for (let elapsed = 1000; elapsed <= 5000; elapsed += 100) sample(elapsed);
  for (let elapsed = 5100; elapsed <= 6000; elapsed += 100) assert.equal(sample(elapsed, 0, 0), false);
  for (let elapsed = 6100; elapsed <= 20000; elapsed += 100) assert.equal(sample(elapsed), false);
  assert.equal(c.submissions, 0);
});

test("quiet speech above the silence threshold does not count as a long pause", () => {
  const { context: c, sample } = alwaysStopContext();
  for (let elapsed = 1000; elapsed <= 2000; elapsed += 100) sample(elapsed);
  for (let elapsed = 2100; elapsed <= 5000; elapsed += 100) assert.equal(sample(elapsed, 40, 6), false);
  assert.equal(sample(5100, 0, 0), false);
  assert.equal(c.submissions, 0);
});

test("Always sends natural speech with short unvoiced gaps after 1.4 seconds of silence", () => {
  const { context: c, sample } = alwaysStopContext();
  for (let elapsed = 10; elapsed <= 1800; elapsed += 10) {
    const voiced = elapsed % 100 < 70;
    assert.equal(sample(elapsed, voiced ? 100 : 0, voiced ? 20 : 0, 10), false);
  }
  for (let elapsed = 1810; elapsed <= 3300; elapsed += 10) sample(elapsed, 0, 0, 10);
  assert.equal(c.submissions, 1);
});

test("a short answer at recording startup sends without requiring another utterance", () => {
  const { context: c, sample } = alwaysStopContext();
  for (let elapsed = 10; elapsed <= 300; elapsed += 10) sample(elapsed, 100, 20, 10);
  for (let elapsed = 310; elapsed <= 2300; elapsed += 10) sample(elapsed, 0, 0, 10);
  assert.equal(c.submissions, 1);
});

test("a brief microphone click alone does not submit a human turn", () => {
  const { context: c, sample } = alwaysStopContext();
  for (let elapsed = 10; elapsed <= 6000; elapsed += 10) {
    const click = elapsed >= 1500 && elapsed <= 1520;
    sample(elapsed, click ? 100 : 0, click ? 20 : 0, 10);
  }
  assert.equal(c.submissions, 0);
});

test("Always declares that the browser owns turn detection", async () => {
  const input = inputContext();
  const socket = await connect(input, 2);
  assert.equal(socket.sent[0].recording_mode, "browser_vad");
});

function pcmStopContext() {
  const { context: c } = alwaysStopContext();
  Object.assign(c, {
    humanCaptureBuffers: [], humanCaptureSampleRate: 24000,
    humanCaptureStreamedBufferCount: 0, humanCaptureFrameCount: 0,
    humanCapturePeak: 0, humanCaptureRms: 0,
    humanInputGainValue: () => 1,
    sent: [],
    sendHumanAudioStreamChunk(pcm) { c.sent.push(pcm); return c.sent.length - 1; },
  });
  loadFunctions(c, "function primeHumanCaptureWithPreRoll(", "function flushCapturedHumanPcmCaptureToStream(");
  loadFunctions(c, "function floatSignalStats(", "function pcmSignalStats(");
  loadFunctions(c, "function encodeHumanFloatBuffersPcm(", "function pcmHasSignal(");
  return c;
}

function pcmWindows(count, level) {
  return Array.from({ length: count }, () => new Float32Array(240).fill(level));
}

test("speech captured before stream acceptance sends after its trailing silence", () => {
  const c = pcmStopContext();
  c.humanStreamingReady = false;
  c.humanCaptureBuffers = pcmWindows(200, 0); // live audio during connection
  const preRoll = pcmWindows(30, 0.1); // preceding short answer
  c.primeHumanCaptureWithPreRoll({ buffers: preRoll, sampleRate: 24000 });
  c.flushPendingHumanPcmCaptureToStream();
  assert.equal(c.submissions, 0);
  assert.equal(c.sent.length, 0);
  c.humanStreamingReady = true;
  c.flushPendingHumanPcmCaptureToStream();
  assert.equal(c.submissions, 1);
  assert.equal(c.humanAlwaysAutoStopAudioMs, 2300);
  assert.equal(new Int16Array(c.sent[0].buffer)[0] > 0, true);
  c.flushPendingHumanPcmCaptureToStream();
  assert.equal(c.sent.length, 1);
});

test("queued audio is checked through its latest speech before deciding to submit", () => {
  const c = pcmStopContext();
  c.humanCaptureBuffers = [
    ...pcmWindows(30, 0.1), ...pcmWindows(250, 0), ...pcmWindows(30, 0.1),
  ];
  c.flushPendingHumanPcmCaptureToStream();
  assert.equal(c.submissions, 0);
  c.humanCaptureBuffers.push(...pcmWindows(140, 0));
  c.flushPendingHumanPcmCaptureToStream();
  assert.equal(c.submissions, 1);
});

test("unsent PCM cannot claim speech detection or advance the audio cursor", () => {
  const c = pcmStopContext();
  c.humanCaptureBuffers = [...pcmWindows(30, 0.1), ...pcmWindows(200, 0)];
  c.sendHumanAudioStreamChunk = () => -1;
  c.flushPendingHumanPcmCaptureToStream();
  assert.equal(c.submissions, 0);
  assert.equal(c.humanCaptureStreamedBufferCount, 0);
  assert.equal(c.humanAlwaysAutoStopAudioMs, 0);
});

test("finishing a pre-roll answer uses captured duration despite a recent connection", async () => {
  const input = inputContext();
  const c = input.context;
  const socket = await connect(input, 2);
  c.runtimeIsHumanClientMode = () => true;
  c.performance.now = () => c.humanRecordingStartedAt + 100;
  c.humanAlwaysAutoStopAudioMs = 2300;
  c.sendHumanAudioStreamChunk(new Uint8Array([1, 0]));
  await c.stopHumanRecordingAndSubmit();
  assert.equal(socket.sent.at(-1).type, "end");
  assert.equal(c.statuses.includes("hold longer to talk"), false);
  assert.equal(c.statuses.at(-1).includes("文字起こし"), true);
});

function bargeInPcmContext({ recording = true, starting = false, gain = 1 } = {}) {
  let now = 10000;
  const c = {
    Date: { now: () => now },
    humanAlwaysRecordingEnabled: true, humanRecording: recording,
    humanRecordingStarting: starting, humanRecordingStopping: false,
    humanStreamingReady: false, humanStreamingSocket: null,
    humanMediaRecorderFallbackActive: false,
    humanAlwaysBargeInControllerActive: false,
    humanAlwaysBargeInSignalActiveMs: 0, humanAlwaysBargeInSignalElapsedMs: 0,
    humanAlwaysBargeInSignalQuietMs: 0, humanAlwaysBargeInLastStartedAt: 0,
    humanAlwaysVadRestartPending: false,
    humanInputPaused: () => false, runtimeStartRequested: true, warmupActive: false,
    activeSources: new Set([{}]),
    humanAlwaysSensitivitySettings: () => ({ peak: 65, peakRmsFloor: 4, rms: 7, requiredWindows: 3, minActiveMs: 500 }),
    alwaysInputModeForCurrentRuntimeState: () => "barge_in",
    resetHumanAlwaysRestartState() {}, setHumanStatus() {}, args: {},
    humanCaptureNode: { port: {} }, humanCaptureSampleRate: 24000,
    humanCaptureBuffers: [], humanCaptureFrameCount: 0, humanCapturePeak: 0, humanCaptureRms: 0,
    normalizeHumanCaptureMessage: data => data,
    humanInputGainValue: () => gain,
    preRoll: [], appendHumanAlwaysMonitorPreRoll(buffer) { c.preRoll.push(buffer); },
    flushed: 0, flushPendingHumanPcmCaptureToStream() { c.flushed += 1; },
    interruptions: 0, started: 0,
    async interruptActiveMainPlayback() { c.interruptions += 1; c.activeSources.clear(); return true; },
    async startAlwaysHumanInputFromMonitor() { c.started += 1; c.humanAlwaysBargeInControllerActive = true; },
  };
  for (const [, name, value] of html.matchAll(/const (HUMAN_ALWAYS_BARGE_IN_[A-Z_]+) = (\d+);/g)) c[name] = Number(value);
  loadFunctions(c, "function resetHumanAlwaysBargeInDetection()", "function resetHumanAlwaysBargeInControllerState()");
  loadFunctions(c, "function canStartAlwaysInputFromLevel()", "async function startAlwaysHumanInputFromMonitor(");
  loadFunctions(c, "function floatSignalStats(", "function pcmSignalStats(");
  const first = html.indexOf("humanCaptureNode.port.onmessage = (event) =>");
  const last = html.indexOf("humanCaptureSilenceGain = humanCaptureContext.createGain();", first);
  vm.runInNewContext(html.slice(first, last), c);
  return {
    context: c,
    sample(level, durationMs = 10) {
      now += durationMs;
      c.humanCaptureNode.port.onmessage({ data: {
        buffer: new Float32Array(Math.round(durationMs * 24)).fill(level),
      } });
    },
  };
}

for (const starting of [false, true]) {
  test(`live PCM interrupts AI playback during recording (connection opening: ${starting})`, () => {
    const { context: c, sample } = bargeInPcmContext({ starting });
    for (let ms = 0; ms < 800; ms += 10) sample(ms % 150 < 100 ? 0.03 : 0);
    assert.equal(c.interruptions, 1);
    assert.equal(c.activeSources.size, 0);
    assert.equal(c.humanRecording, true);
    assert.equal(c.humanCaptureBuffers.length, 80);
    assert.equal(c.flushed, 0); // recording survives while the socket opens
  });
}

test("idle Always PCM detects barge-in and retains speech before opening the input", () => {
  const { context: c, sample } = bargeInPcmContext({ recording: false });
  for (let ms = 0; ms < 800; ms += 10) sample(ms % 150 < 100 ? 0.03 : 0);
  assert.equal(c.started, 1);
  assert.equal(c.preRoll.length, 80);
});

test("PCM barge-in applies the same microphone gain as speech submission", () => {
  const { context: c, sample } = bargeInPcmContext({ gain: 4 });
  for (let ms = 0; ms < 800; ms += 10) sample(0.003);
  assert.equal(c.interruptions, 1);
});

test("silence and isolated clicks do not interrupt playback", () => {
  const { context: c, sample } = bargeInPcmContext();
  for (let ms = 0; ms < 2000; ms += 10) sample(ms % 400 < 20 ? 0.1 : 0);
  assert.equal(c.interruptions, 0);
});

test("paused human input cannot interrupt playback", () => {
  const { context: c, sample } = bargeInPcmContext();
  c.humanInputPaused = () => true;
  for (let ms = 0; ms < 800; ms += 10) sample(0.03);
  assert.equal(c.interruptions, 0);
});

test("a silent diagnostic meter cannot erase the PCM speech used for interruption", () => {
  const { context: c, sample } = bargeInPcmContext();
  const node = { connect() {}, gain: { value: 0 } };
  c.window = {
    AudioContext: class {
      createMediaStreamSource() { return node; }
      createGain() { return node; }
      createAnalyser() { return { ...node, getFloatTimeDomainData(data) { data.fill(0); } }; }
      async resume() {}
    },
    setInterval(callback) { c.meterTick = callback; return 1; },
  };
  c.humanRecordingStartedAt = 0;
  c.HUMAN_ALWAYS_NO_SIGNAL_WARNING_MS = 3000;
  c.stopHumanLevelMeter = c.updateHumanInputLevel = () => {};
  c.humanCaptureDiagnostic = c.humanMicrophoneTrackDiagnostic = () => "";
  loadFunctions(c, "function startHumanLevelMeter(stream)", "function stopHumanLevelMeter()");
  c.startHumanLevelMeter({});
  assert.equal(typeof c.meterTick, "function");
  for (let ms = 0; ms < 800; ms += 10) {
    sample(ms % 150 < 100 ? 0.03 : 0);
    if (ms % 250 === 0) c.meterTick();
  }
  assert.equal(c.interruptions, 1);
});

test("PCM interruption stops current and queued AI audio before the server responds", async () => {
  const { context: c, sample } = bargeInPcmContext({ starting: true });
  const metadata = { key: "ai-turn", speaker: "counselor", turn_id: 5 };
  const stopped = [];
  const segments = [
    { key: "ai-turn", metadata, startAt: 1, endAt: 8, source: { stop() { stopped.push("current"); } } },
    { key: "ai-turn", metadata, startAt: 8, endAt: 15, source: { stop() { stopped.push("queued"); } } },
  ];
  c.audioContext = { currentTime: 4 };
  c.mainPlaybackSegments = new Map([["ai-turn", new Set(segments)]]);
  c.activeSources = new Set(segments.map(s => s.source));
  c.interruptedPlaybackKeys = new Set();
  c.mainPlaybackProgressByKey = new Map([["ai-turn", segments]]);
  c.PLAYBACK_LANE_MAIN = "main";
  c.playbackLanes = { main: { scheduledTime: 15, pendingPcmByteLength: 100 } };
  c.playbackKeyFromMetadata = m => m.key;
  c.playbackLaneForMetadata = () => c.playbackLanes.main;
  c.renderTimeline = c.setStatus = () => {};
  c.args.runtime_interrupt_url = "/runtime/interrupt";
  const requests = [];
  let reply;
  c.postRuntimeControl = (url, message, payload) => {
    requests.push(payload);
    return new Promise(resolve => { reply = resolve; });
  };
  loadFunctions(c, "function removeMainPlaybackSegment(segment)", "function trackMainPlaybackSegment(");
  loadFunctions(c, "function playedSecondsForInterruptedSegment(segment)", "function emitPlaybackCompletedIfReady(");
  loadFunctions(c, "function interruptPayloadForSegment(", "function timelineItemForMetadata(");
  loadFunctions(c, "function clearAllPcmRemainders()", "function resetPlaybackLanes()");
  loadFunctions(c, "function clearPlaybackBatch(lane)", "function playbackBatchTargetByteLength(");
  for (let ms = 0; ms < 800; ms += 10) sample(ms % 150 < 100 ? 0.03 : 0);
  assert.deepEqual(stopped, ["current", "queued"]);
  assert.equal(c.activeSources.size, 0);
  assert.equal(c.mainPlaybackSegments.size, 0);
  assert.equal(c.playbackLanes.main.pendingPcmByteLength, 0);
  assert.equal(c.humanRecording, true);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].played_ms, 3000);
  assert.equal(requests[0].turn_id, 5);
  assert.equal(requests[0].reason, "human_barge_in");

  // Audio already in flight must not resume the interrupted turn.
  c.audioPlaybackPaused = false;
  c.updateMetadata = c.appendPcmChunkToPlaybackBatch = () => assert.fail("interrupted audio scheduled");
  loadFunctions(c, "function handleAudioChunkMessage(message)", "function handleTurnEndMessage(");
  assert.equal(c.handleAudioChunkMessage({ metadata }), true);
  reply(true);
  await Promise.resolve();
});

test("an empty interrupted input waits for AI recovery without reopening silent capture", async () => {
  const input = inputContext();
  const c = input.context;
  c.runtimeIsHumanClientMode = () => true;
  const socket = await connect(input, 2);
  socket.emit("message", { type: "completed", completion_reason: "blank_transcript",
    resume_pending: true, resume_after_ms: 3000, text: "" });
  assert.equal(c.humanAlwaysAiResumePending, true);
  assert.equal(c.humanStreamingError, "");
  c.humanLevelAnalyser = {};
  c.humanAlwaysBargeInControllerActive = false;
  c.startAlwaysBargeInCaptureMonitor = async () => true;
  c.startHumanRecording = async () => assert.fail("silent capture reopened during recovery");
  loadFunctions(c, "async function maybeResumeAlwaysHumanRecordingFromStatus(status)", "function humanMicrophoneTrackDiagnostic()");
  const status = { phase: "running", awaiting_human_input: true, human_input_state: "waiting_to_resume_interrupted_ai" };
  await c.maybeResumeAlwaysHumanRecordingFromStatus(status);
  assert.match(c.alwaysArmedStatusForRuntimeStatus(status), /AI.*再開/);
  // A stale input-wait status must not cancel the announced recovery either.
  await c.maybeResumeAlwaysHumanRecordingFromStatus({ phase: "running", awaiting_human_input: true });
  c.cleanupHumanRecording();
  assert.equal(c.humanAlwaysAiResumePending, false);
});

test("false barge-in waits for actual silence even with quiet ongoing speech", () => {
  const { context: c, sample } = alwaysStopContext();
  c.humanRecordingStartedByBargeIn = true;
  for (let ms = 10; ms <= 4000; ms += 10) sample(ms, 40, 6, 10);
  assert.equal(c.falseStarts, 0);
  for (let ms = 4010; ms <= 5400; ms += 10) sample(ms, 0, 0, 10);
  assert.equal(c.falseStarts, 1);
});

test("new microphone input clears the UI recovery wait", async () => {
  const input = inputContext();
  input.context.humanAlwaysAiResumePending = true;
  const starting = input.context.startHumanRecording();
  for (let i = 0; i < 30 && input.sockets.length === 0; i += 1) await Promise.resolve();
  const socket = input.sockets.at(-1);
  assert.ok(socket);
  socket.emit("open");
  socket.emit("message", { type: "accepted", turn_id: 2 });
  await starting;
  assert.equal(input.context.humanAlwaysAiResumePending, false);
  assert.equal(input.context.humanRecording, true);
});
