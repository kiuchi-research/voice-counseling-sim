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
