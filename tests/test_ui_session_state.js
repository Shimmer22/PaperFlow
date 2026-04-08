const assert = require('assert');
const {
  normalizeIdeaText,
  shouldResetClarificationSession,
} = require('../ui/src/session_state');

assert.strictEqual(normalizeIdeaText('  波束成形\n和自注意力  '), '波束成形 和自注意力');
assert.strictEqual(shouldResetClarificationSession('旧想法', '新想法', 1), true);
assert.strictEqual(shouldResetClarificationSession('同一个想法', '同一个想法', 2), false);
assert.strictEqual(shouldResetClarificationSession('', '新想法', 0), false);

console.log('ui session state tests passed');
