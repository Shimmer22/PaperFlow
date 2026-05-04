const assert = require('assert');
const {
  buildProgressLines,
  effectiveProgressPercent,
  progressPercentLabel,
} = require('../ui/src/run_progress_state');

assert.strictEqual(progressPercentLabel({ progressPercent: 37 }), '37%');
assert.strictEqual(progressPercentLabel({}), '0%');
assert.strictEqual(
  effectiveProgressPercent(
    {
      status: 'running',
      progressPercent: 12,
      stageStartedAt: '2026-05-04T00:00:00.000Z',
      stageEtaSeconds: 100,
      stagePercentMin: 12,
      stagePercentMax: 21,
    },
    Date.parse('2026-05-04T00:00:50.000Z'),
  ),
  17,
);
assert.strictEqual(
  progressPercentLabel(
    {
      status: 'running',
      progressPercent: 12,
      stageStartedAt: '2026-05-04T00:00:00.000Z',
      stageEtaSeconds: 100,
      stagePercentMin: 12,
      stagePercentMax: 21,
    },
    Date.parse('2026-05-04T00:00:50.000Z'),
  ),
  '17%',
);

const lines = buildProgressLines({
  status: 'running',
  stageLabel: '初筛候选',
  progressPercent: 62,
  message: '正在评估第 8 / 13 篇候选论文',
  progressDetail: '8 / 13',
});

assert.ok(lines.some((line) => line.includes('状态：运行中')));
assert.ok(lines.some((line) => line.includes('阶段：初筛候选')));
assert.ok(lines.some((line) => line.includes('进度：62%')));
assert.ok(lines.some((line) => line.includes('当前：8 / 13')));

console.log('run progress state tests passed');
