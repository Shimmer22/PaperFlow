(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
  root.RunProgressState = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  function clampPercent(value) {
    return Math.max(0, Math.min(100, Math.round(Number(value || 0))));
  }

  function effectiveProgressPercent(state, nowMs) {
    const base = clampPercent(state?.progressPercent || 0);
    if ((state?.status || '') !== 'running') {
      return base;
    }
    const startedAt = String(state?.stageStartedAt || '').trim();
    const etaSeconds = Number(state?.stageEtaSeconds || 0);
    const stageMin = clampPercent(state?.stagePercentMin ?? base);
    const stageMax = clampPercent(state?.stagePercentMax ?? base);
    if (!startedAt || etaSeconds <= 0 || stageMax <= stageMin) {
      return base;
    }
    const startedMs = Date.parse(startedAt);
    if (!Number.isFinite(startedMs)) {
      return base;
    }
    const currentMs = Number.isFinite(nowMs) ? nowMs : Date.now();
    const elapsedSeconds = Math.max(0, (currentMs - startedMs) / 1000);
    const ratio = Math.min(0.9, elapsedSeconds / etaSeconds);
    const interpolated = stageMin + (stageMax - stageMin) * ratio;
    return Math.max(base, clampPercent(interpolated));
  }

  function statusLabel(status) {
    if (status === 'running') return '运行中';
    if (status === 'completed') return '已完成';
    if (status === 'completed_with_warnings') return '完成但有警告';
    if (status === 'failed') return '失败';
    return status || '未知';
  }

  function progressPercentLabel(state, nowMs) {
    return `${effectiveProgressPercent(state, nowMs)}%`;
  }

  function buildProgressLines(state) {
    if (!state) {
      return ['暂无运行中的任务。'];
    }
    return [
      `状态：${statusLabel(state.status)}`,
      state.stageLabel ? `阶段：${state.stageLabel}` : '',
      `进度：${progressPercentLabel(state)}`,
      state.progressDetail ? `当前：${state.progressDetail}` : '',
      state.providerName ? `Provider：${state.providerName}` : '',
      state.mainModel ? `主模型：${state.mainModel}` : '',
      state.mainReasoningEffort ? `主思考强度：${state.mainReasoningEffort}` : '',
      state.subModel ? `Sub 模型：${state.subModel}` : '',
      state.subReasoningEffort ? `Sub 思考强度：${state.subReasoningEffort}` : '',
      state.outdir ? `输出目录：${state.outdir}` : '',
      state.message ? `说明：${state.message}` : '',
    ].filter(Boolean);
  }

  return {
    statusLabel,
    effectiveProgressPercent,
    progressPercentLabel,
    buildProgressLines,
  };
});
