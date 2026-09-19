const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const html = fs.readFileSync(
  path.join(__dirname, "../app/components/runtime_audio_monitor/index.html"),
  "utf8",
);
const start = html.indexOf("function updateMainPlaybackProgress(");
const end = html.indexOf("function emitPlaybackCompletedIfReady(", start);
assert.ok(start >= 0 && end > start);

function playbackContext() {
  const context = {
    audioContext: { currentTime: 0 },
    mainPlaybackProgressByKey: new Map(),
    playbackKeyFromMetadata: (metadata) => metadata.key,
  };
  vm.runInNewContext(html.slice(start, end), context);
  return context;
}

test("interruption position excludes gaps between received audio batches", () => {
  const context = playbackContext();
  context.updateMainPlaybackProgress("turn", { startAt: 0, endAt: 8 });
  context.updateMainPlaybackProgress("turn", { startAt: 9.13, endAt: 17.78 });
  context.audioContext.currentTime = 17.78;
  const played = context.playedSecondsForInterruptedSegment({ key: "turn" });
  assert.ok(Math.abs(played - 16.65) < 1e-9, `received ${played}`);
});

test("position remains at the end of received audio during a scheduling gap", () => {
  const context = playbackContext();
  context.updateMainPlaybackProgress("turn", { startAt: 10, endAt: 11 });
  context.updateMainPlaybackProgress("turn", { startAt: 12, endAt: 13 });
  context.audioContext.currentTime = 11.5;
  assert.equal(context.playedSecondsForInterruptedSegment({ key: "turn" }), 1);
  context.audioContext.currentTime = 12.25;
  assert.equal(context.playedSecondsForInterruptedSegment({ key: "turn" }), 1.25);
});

test("future audio and audio from other turns are not counted as played", () => {
  const context = playbackContext();
  context.updateMainPlaybackProgress("turn", { startAt: 10, endAt: 11 });
  context.updateMainPlaybackProgress("other", { startAt: 20, endAt: 30 });
  context.audioContext.currentTime = 9;
  assert.equal(context.playedSecondsForInterruptedSegment({ key: "turn" }), 0);
  context.audioContext.currentTime = 100;
  assert.equal(context.playedSecondsForInterruptedSegment({ key: "turn" }), 1);
});

test("a standalone playback segment has a bounded audio position", () => {
  const context = playbackContext();
  const segment = { startAt: 2, endAt: 3 };
  context.audioContext.currentTime = 2.25;
  assert.equal(context.playedSecondsForInterruptedSegment(segment), 0.25);
  context.audioContext.currentTime = 4;
  assert.equal(context.playedSecondsForInterruptedSegment(segment), 1);
});

function warmupContext() {
  const timers = new Map();
  const played = [];
  let now = 0;
  let nextTimerId = 0;
  const context = {
    window: {
      RUNTIME_AUDIO_MONITOR_ARGS: { warmup_target_turns: 7 },
      setTimeout(callback, ms) {
        const id = ++nextTimerId;
        timers.set(id, { callback, deadline: now + ms });
        return id;
      },
      clearTimeout(id) { timers.delete(id); },
    },
    Date: { now: () => now },
    warmupActive: false,
    warmupStartedAt: 0,
    warmupTimer: null,
    warmupBufferedMessages: [],
    warmupCompletedTurnKeys: new Set(),
    warmupIndicatorEl: { classList: { add() {}, remove() {} } },
    warmupTextEl: { textContent: "" },
    setFrameHeight() {},
    setStatus() {},
    updateMetadata() {},
    playbackKeyFromMetadata: (metadata) => metadata.key,
    audioPlaybackPaused: false,
    audioContext: { state: "running", currentTime: 0 },
    pausedPlaybackMessages: [],
    resetPlaybackLaneSchedules() {},
    refreshTimelineTimer() {},
    updateGain() {},
    updateHumanInputVisibility() {},
    stopRuntimeInlineStatusPolling() {},
    setRuntimeInlineState() {},
    isMonitorArmed: () => false,
    handlePlaybackMessage(message) { played.push(message); return true; },
    advance(ms) {
      now += ms;
      for (const [id, timer] of timers) {
        if (timer.deadline <= now) {
          timers.delete(id);
          timer.callback();
        }
      }
    },
    played,
  };
  for (const [first, next] of [
    ["let args = {", "let socket = null;"],
    ["function numericArg(", "function ensureAudioContext("],
    ["function flushPausedPlaybackMessages(", "function timelineKeyFromMetadata("],
    ["function flushWarmupPlaybackBuffer(", "function updateHumanInputVisibility("],
    ["function applyArgs(", "startButton.addEventListener("],
  ]) {
    const from = html.indexOf(first);
    const to = html.indexOf(next, from);
    assert.ok(from >= 0 && to > from, first);
    vm.runInNewContext(html.slice(from, to), context);
  }
  return context;
}

test("seven-turn warmup waits beyond 24 seconds and plays all turns in order", () => {
  const context = warmupContext();
  context.beginWarmupBuffering();
  const expected = [];
  for (let turn = 0; turn < 7; turn += 1) {
    const audio = { type: "audio_chunk", metadata: { key: `turn-${turn}` } };
    const end = { type: "turn_end", metadata: audio.metadata };
    expected.push(audio, end);
    assert.equal(context.maybeBufferWarmupMessage(audio), true);
    context.advance(30000);
    assert.equal(context.played.length, 0, `audio played before turn ${turn + 1} finished`);
    assert.equal(context.maybeBufferWarmupMessage(end), true);
    assert.equal(context.warmupActive, turn < 6);
  }
  assert.deepEqual(context.played, expected);
});

test("short sessions release buffered audio when the stream ends", () => {
  const context = warmupContext();
  context.beginWarmupBuffering();
  const audio = { type: "audio_chunk", metadata: { key: "turn-0" } };
  const end = { type: "turn_end", metadata: audio.metadata };
  context.maybeBufferWarmupMessage(audio);
  context.maybeBufferWarmupMessage(end);
  context.flushWarmupPlaybackBuffer("stream_end");
  assert.equal(context.warmupActive, false);
  assert.deepEqual(context.played, [audio, end]);
});

test("resuming during warmup keeps pending audio buffered until seven turns finish", () => {
  const context = warmupContext();
  context.beginWarmupBuffering();
  const expected = [];
  for (let turn = 0; turn < 7; turn += 1) {
    const audio = { type: "audio_chunk", metadata: { key: `turn-${turn}` } };
    const end = { type: "turn_end", metadata: audio.metadata };
    expected.push(audio, end);
    context.pausedPlaybackMessages.push(audio, end);
    assert.equal(context.flushPausedPlaybackMessages(), true);
    assert.equal(context.played.length, turn < 6 ? 0 : 14);
  }
  assert.deepEqual(context.played, expected);
});

test("lowering the target during warmup releases already sufficient buffered turns", () => {
  const context = warmupContext();
  context.beginWarmupBuffering();
  for (let turn = 0; turn < 6; turn += 1) {
    const metadata = { key: `turn-${turn}` };
    context.maybeBufferWarmupMessage({ type: "audio_chunk", metadata });
    context.maybeBufferWarmupMessage({ type: "turn_end", metadata });
  }
  context.applyArgs({ warmup_target_turns: 5 });
  assert.equal(context.warmupActive, false);
  assert.equal(context.played.length, 12);
});
