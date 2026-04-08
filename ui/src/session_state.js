function normalizeIdeaText(text) {
  return String(text || '').replace(/\s+/g, ' ').trim();
}

function shouldResetClarificationSession(previousIdea, currentIdea, historyLength) {
  if (!historyLength) {
    return false;
  }
  return normalizeIdeaText(previousIdea) !== normalizeIdeaText(currentIdea);
}

module.exports = {
  normalizeIdeaText,
  shouldResetClarificationSession,
};
