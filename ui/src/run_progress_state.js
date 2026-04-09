(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
  root.RunProgressState = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  function statusLabel(status) {
    if (status === 'running') return '运行中';
    if (status === 'completed') return '已完成';
    if (status === 'completed_with_warnings') return '完成但有警告';
    if (status === 'failed') return '失败';
    return status || '未知';
  }

  function progressPercentLabel(state) {
    const value = Number(state?.progressPercent || 0);
    return `${Math.max(0, Math.min(100, Math.round(value)))}%`;
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
    progressPercentLabel,
    buildProgressLines,
  };
});
