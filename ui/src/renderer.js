const runsList = document.getElementById('runsList');
const outputRootInput = document.getElementById('outputRoot');
const refreshRunsBtn = document.getElementById('refreshRuns');
const openRunDirBtn = document.getElementById('openRunDir');
const openRunLogBtn = document.getElementById('openRunLog');
const openRunLogInlineBtn = document.getElementById('openRunLogInline');
const providerSelect = document.getElementById('providerSelect');
const mainModelSelect = document.getElementById('mainModelSelect');
const mainModelInput = document.getElementById('mainModelInput');
const mainReasoningSelect = document.getElementById('mainReasoningSelect');
const subModelSelect = document.getElementById('subModelSelect');
const subModelInput = document.getElementById('subModelInput');
const subReasoningSelect = document.getElementById('subReasoningSelect');
const candidateLimitInput = document.getElementById('candidateLimit');
const ideaInput = document.getElementById('ideaInput');
const maxPapersInput = document.getElementById('maxPapers');
const sourcesInput = document.getElementById('sourcesInput');
const generateClarificationTurnBtn = document.getElementById('generateClarificationTurn');
const clarificationQuestion = document.getElementById('clarificationQuestion');
const clarificationOptions = document.getElementById('clarificationOptions');
const clarificationNote = document.getElementById('clarificationNote');
const appendClarificationAnswerBtn = document.getElementById('appendClarificationAnswer');
const outdirInput = document.getElementById('outdirInput');
const downloadPdfInput = document.getElementById('downloadPdf');
const parallelRunInput = document.getElementById('parallelRun');
const startRunBtn = document.getElementById('startRun');
const currentRunStatus = document.getElementById('currentRunStatus');
const runSummary = document.getElementById('runSummary');
const candidates = document.getElementById('candidates');
const selected = document.getElementById('selected');
const briefs = document.getElementById('briefs');
const finalDiscussion = document.getElementById('finalDiscussion');

let currentRunDir = null;
let providerConfigs = [];
let currentRunState = null;
let currentLoadedRun = null;
let clarificationHistory = [];
let currentClarificationTurn = null;
let selectedClarificationOptionId = '';
let clarificationIdea = '';

function normalizeIdeaText(text) {
  return String(text || '').replace(/\s+/g, ' ').trim();
}

function shouldResetClarificationSession(previousIdea, currentIdea, historyLength) {
  if (!historyLength) {
    return false;
  }
  return normalizeIdeaText(previousIdea) !== normalizeIdeaText(currentIdea);
}

function escapeHtml(text) {
  return String(text || '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function conciseWarningMessage(warningText) {
  const text = String(warningText || '');
  if (!text) return '';
  if (text.includes('触发速率限制')) {
    return text;
  }
  if (text.includes('鉴权失败')) {
    return text;
  }
  if (text.includes('模型不可用') || text.includes('EOL') || text.includes('410')) {
    return text;
  }
  if (text.includes('429')) {
    if (text.toLowerCase().includes('openalex')) {
      return 'OpenAlex 暂时限流了这次查询，请稍后重试或降低查询频率。';
    }
    if (text.toLowerCase().includes('semanticscholar')) {
      return 'Semantic Scholar 暂时限流了这次查询，请稍后重试。';
    }
    if (text.toLowerCase().includes('nvidia')) {
      return 'NVIDIA API 触发了限流，请稍后重试，或切换到其他 provider。';
    }
    if (text.toLowerCase().includes('glm')) {
      return 'GLM API 触发了限流，请稍后重试，或降低操作频率。';
    }
    return '接口触发了限流，请稍后重试。';
  }
  if (text.includes('301')) {
    return '论文源发生了重定向，当前查询可能没有被正确处理。';
  }
  if (text.includes('nodename nor servname') || text.includes('Name or service not known')) {
    return '当前网络或 DNS 不可用，论文源无法连接。';
  }
  if (text.includes('provider exited')) {
    return 'CLI provider 返回失败，模型阶段没有产出结构化结果。';
  }
  if (text.includes('provider timed out')) {
    return 'CLI provider 超时，主模型阶段已退回本地规则。';
  }
  return '本次运行中有一步失败，建议打开日志查看详细原因。';
}

function stageLabel(value) {
  if (value === 'provider_success') return '主模型完成';
  if (value === 'provider_failed_local_fallback') return '主模型失败，已退回本地';
  if (value === 'provider_unavailable_local_fallback') return '主模型不可用，已退回本地';
  if (value === 'local_only') return '本地规则';
  if (value === 'completed') return '完成';
  if (value === 'completed_with_partial_source_failures') return '完成，但部分论文源失败';
  if (value === 'no_candidates_with_source_failures') return '无候选，且伴随论文源故障';
  if (value === 'no_candidates') return '无候选';
  return value || '-';
}

function scoreExplanation(item) {
  const score = item.score_breakdown || {};
  const parts = [];
  if ((score.idea_relevance ?? 0) >= 0.7) {
    parts.push('和你的研究想法高度相关');
  } else if ((score.idea_relevance ?? 0) >= 0.45) {
    parts.push('和你的研究想法有一定相关性');
  } else {
    parts.push('和你的研究想法只有较弱相关性');
  }
  if ((score.method_relevance ?? 0) >= 0.45) {
    parts.push('方法路线也比较贴近');
  } else if ((score.method_relevance ?? 0) >= 0.15) {
    parts.push('方法上有部分重叠');
  } else {
    parts.push('但方法路线贴合度不高');
  }
  if ((score.importance ?? 0) >= 0.7) {
    parts.push('同时具备较高影响力或关注度');
  }
  if ((score.evidence_quality ?? 0) >= 0.8) {
    parts.push('元数据和摘要信息比较完整');
  }
  return `${item.selected ? '入选原因' : '排序判断'}：${parts.join('，')}。`;
}

function renderCards(container, items, format) {
  container.innerHTML = '';
  for (const item of items || []) {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = format(item);
    container.appendChild(card);
  }
}

async function refreshRuns() {
  const outputRoot = outputRootInput.value;
  const runs = await window.researchFlow.listRuns(outputRoot);
  runsList.innerHTML = '';
  for (const run of runs) {
    const item = document.createElement('div');
    item.className = 'history-item';

    const title = document.createElement('div');
    title.className = 'history-title';
    title.contentEditable = 'true';
    title.spellcheck = false;
    title.textContent = run.alias || run.summary.run_id;
    title.addEventListener('click', () => loadRun(run.runDir));
    title.addEventListener('blur', async () => {
      const alias = title.textContent.trim();
      await window.researchFlow.saveRunAlias(run.runDir, alias === run.summary.run_id ? '' : alias);
      await refreshRuns();
    });

    const meta = document.createElement('div');
    meta.className = 'history-meta';
    meta.textContent = `${run.summary.status} · 候选 ${run.summary.candidate_count} · 入选 ${run.summary.selected_count}`;
    meta.addEventListener('click', () => loadRun(run.runDir));

    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'delete-run-btn';
    deleteBtn.textContent = '×';
    deleteBtn.title = '删除这次运行';
    deleteBtn.addEventListener('click', async (event) => {
      event.stopPropagation();
      const ok = window.confirm(`确定删除这次运行吗？\n${run.alias || run.summary.run_id}`);
      if (!ok) return;
      await window.researchFlow.deleteRun(run.runDir);
      if (currentRunDir === run.runDir) {
        currentRunDir = null;
        currentLoadedRun = null;
      }
      await refreshRuns();
    });

    item.appendChild(deleteBtn);
    item.appendChild(title);
    item.appendChild(meta);
    runsList.appendChild(item);
  }
}

async function loadRun(runDir) {
  currentRunDir = runDir;
  const data = await window.researchFlow.loadRun(runDir);
  currentLoadedRun = data;
  renderRunSummary(data.runSummary || {}, data.runManifest || {});
  renderCards(candidates, data.candidates || [], (item) => `
    <h3>${item.paper.title}</h3>
    <p>总分: ${item.score_breakdown.total}</p>
    <p>${scoreExplanation(item)}</p>
    ${item.rejection_reason ? `<p><strong>未入选说明:</strong> ${escapeHtml(item.rejection_reason)}</p>` : ''}
  `);
  renderCards(selected, data.selected || [], (item) => `
    <h3>${item.title}</h3>
    <p>${item.venue || '未知 venue'} ${item.year || ''}</p>
    <p class="break-text">${item.url ? `<a href="${item.url}">${item.url}</a>` : ''}</p>
  `);
  renderCards(briefs, data.briefs || [], (item) => `
    <h3>${item.bibliographic_info.title}</h3>
    <p>${item.one_sentence_summary}</p>
    <p><strong>核心方法:</strong> ${item.core_method}</p>
    <p><strong>主要结果:</strong> ${((item.main_results || [])[0]) || '暂无明确结果摘要。'}</p>
    <p><strong>阅读深度:</strong> ${item.reading_depth || 'unknown'}</p>
  `);
  finalDiscussion.textContent = data.finalDiscussion || '';
}

function fillSelect(selectEl, values, placeholder) {
  selectEl.innerHTML = '';
  const first = document.createElement('option');
  first.value = '';
  first.textContent = placeholder;
  selectEl.appendChild(first);
  for (const value of values) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = value;
    selectEl.appendChild(option);
  }
}

function syncProviderOptions() {
  const provider = providerConfigs.find((item) => item.filePath === providerSelect.value);
  fillSelect(mainModelSelect, provider?.supportedModels || [], '默认主模型');
  fillSelect(mainReasoningSelect, provider?.supportedReasoningEfforts || [], '默认主强度');
  fillSelect(subModelSelect, provider?.supportedModels || [], '默认 Sub 模型');
  fillSelect(subReasoningSelect, provider?.supportedReasoningEfforts || [], '默认 Sub 强度');
}

function renderClarificationTurn(turn) {
  currentClarificationTurn = turn || null;
  selectedClarificationOptionId = '';
  clarificationOptions.innerHTML = '';
  if (!turn) {
    clarificationQuestion.textContent = '尚未生成澄清问题。';
    return;
  }
  clarificationQuestion.textContent = `${turn.question || ''}${turn.ready_for_research ? '\n已满足检索条件。' : ''}`;
  for (const opt of turn.options || []) {
    const row = document.createElement('label');
    row.className = 'checkbox';
    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = 'clarification_option';
    radio.value = opt.id;
    radio.addEventListener('change', () => {
      selectedClarificationOptionId = opt.id;
    });
    const text = document.createElement('span');
    text.textContent = `${opt.label}: ${opt.description || ''}`;
    row.appendChild(radio);
    row.appendChild(text);
    clarificationOptions.appendChild(row);
  }
}

function resetClarificationSession(message = '澄清会话已重置。') {
  clarificationHistory = [];
  currentClarificationTurn = null;
  selectedClarificationOptionId = '';
  clarificationIdea = '';
  clarificationOptions.innerHTML = '';
  clarificationNote.value = '';
  clarificationQuestion.textContent = message;
}

async function loadUiConfig() {
  const config = await window.researchFlow.getUiConfig();
  providerConfigs = config.providers || [];
  outputRootInput.value = config.defaultOutputRoot || outputRootInput.value;
  providerSelect.innerHTML = '';
  for (const provider of providerConfigs) {
    const option = document.createElement('option');
    option.value = provider.filePath;
    option.textContent = `${provider.displayName} (${provider.command})`;
    providerSelect.appendChild(option);
  }
  if (providerConfigs[0]) {
    providerSelect.value = providerConfigs[0].filePath;
  }
  syncProviderOptions();
  const runState = await window.researchFlow.getRunState();
  if (runState) {
    currentRunState = runState;
    renderCurrentRunStatus(runState);
  }
}

async function startRun() {
  const provider = providerConfigs.find((item) => item.filePath === providerSelect.value);
  const mainModel = mainModelInput.value.trim() || mainModelSelect.value;
  const mainReasoningEffort = mainReasoningSelect.value;
  const subModel = subModelInput.value.trim() || subModelSelect.value;
  const subReasoningEffort = subReasoningSelect.value;
  const idea = ideaInput.value.trim();
  if (!idea) {
    currentRunStatus.textContent = '请先输入研究想法。';
    return;
  }
  if (shouldResetClarificationSession(clarificationIdea, idea, clarificationHistory.length)) {
    resetClarificationSession('研究想法已变更，旧澄清会话已清空。请先重新生成澄清问题。');
    currentRunStatus.textContent = '研究想法已变更，请重新生成澄清问题后再启动。';
    return;
  }
  currentRunStatus.textContent = '正在启动任务...';
  const outdir = outdirInput.value.trim() || `${outputRootInput.value}/ui_run_${Date.now()}`;
  const result = await window.researchFlow.startRun({
    idea,
    providerName: provider?.name || 'glm_api',
    providerConfigPath: providerSelect.value,
    mainModel,
    mainReasoningEffort,
    subModel,
    subReasoningEffort,
    candidateLimit: Number(candidateLimitInput.value || 5),
    maxPapers: Number(maxPapersInput.value || 5),
    sources: sourcesInput.value.trim(),
    outdir,
    downloadPdf: downloadPdfInput.checked,
    parallel: parallelRunInput.checked,
    clarificationHistory,
  });
  currentRunState = result.run || null;
  renderCurrentRunStatus(currentRunState || { status: 'failed', message: result.error || '启动失败。' });
  if (result.ok) {
    outdirInput.value = outdir;
    await refreshRuns();
  }
}

async function generateClarificationTurn() {
  const provider = providerConfigs.find((item) => item.filePath === providerSelect.value);
  const idea = ideaInput.value.trim();
  if (!idea) {
    clarificationQuestion.textContent = '请先输入研究想法。';
    return;
  }
  if (shouldResetClarificationSession(clarificationIdea, idea, clarificationHistory.length)) {
    resetClarificationSession('检测到研究想法已变更，已清空旧澄清会话。');
  }
  clarificationQuestion.textContent = '澄清问题生成中...';
  const mainModel = mainModelInput.value.trim() || mainModelSelect.value;
  const mainReasoningEffort = mainReasoningSelect.value;
  const result = await window.researchFlow.generateClarificationTurn({
    idea,
    providerConfigPath: provider?.filePath,
    mainModel,
    mainReasoningEffort,
    history: clarificationHistory,
  });
  if (!result.ok) {
    clarificationQuestion.textContent = `生成失败：${result.error || 'unknown error'}`;
    return;
  }
  clarificationIdea = normalizeIdeaText(idea);
  renderClarificationTurn(result.payload?.turn || null);
  if (result.payload?.provider_result && result.payload.provider_result.success === false) {
    clarificationQuestion.textContent += `\n主模型本轮失败，已回退本地澄清：${conciseWarningMessage(result.payload.provider_result.error || 'unknown error')}`;
  }
}

function appendClarificationAnswer() {
  if (!currentClarificationTurn) {
    clarificationQuestion.textContent = '请先生成澄清问题。';
    return;
  }
  if (!selectedClarificationOptionId) {
    clarificationQuestion.textContent = '请先选择一个选项。';
    return;
  }
  clarificationHistory.push({
    idea_snapshot: clarificationIdea || normalizeIdeaText(ideaInput.value),
    question: currentClarificationTurn.question || '',
    selected_option_id: selectedClarificationOptionId,
    user_note: (clarificationNote.value || '').trim(),
    timestamp: Date.now(),
  });
  clarificationNote.value = '';
  clarificationQuestion.textContent = `已记录 ${clarificationHistory.length} 轮澄清。可继续生成下一轮，或直接运行。`;
}

function statusLabel(status) {
  if (status === 'running') return '运行中';
  if (status === 'completed') return '已完成';
  if (status === 'completed_with_warnings') return '完成但有警告';
  if (status === 'failed') return '失败';
  return status || '未知';
}

function renderCurrentRunStatus(state) {
  if (!state) {
    currentRunStatus.textContent = '暂无运行中的任务。';
    return;
  }
  const lines = [
    `状态：${statusLabel(state.status)}`,
    state.providerName ? `Provider：${state.providerName}` : '',
    state.mainModel ? `主模型：${state.mainModel}` : '',
    state.mainReasoningEffort ? `主思考强度：${state.mainReasoningEffort}` : '',
    state.subModel ? `Sub 模型：${state.subModel}` : '',
    state.subReasoningEffort ? `Sub 思考强度：${state.subReasoningEffort}` : '',
    state.outdir ? `输出目录：${state.outdir}` : '',
    state.message ? `说明：${state.message}` : '',
  ].filter(Boolean);
  currentRunStatus.textContent = lines.join('\n');
}

function renderRunSummary(summary, manifest) {
  const warnings = summary.warnings || manifest.warnings || [];
  const keyWarning = warnings[0] || '';
  const runLogPath = manifest?.artifacts?.log || '';
  const keyNotes = summary.key_notes || [];
  const keyNotesHtml = keyNotes.length
    ? `<div class="summary-note"><strong>关键结论：</strong>${keyNotes.slice(0, 3).map((note) => escapeHtml(note)).join('；')}</div>`
    : '';
  runSummary.innerHTML = `
    <div class="summary-grid">
      <div class="summary-item"><span class="summary-label">Run ID</span><span class="summary-value">${escapeHtml(summary.run_id || '-')}</span></div>
      <div class="summary-item"><span class="summary-label">状态</span><span class="summary-value">${escapeHtml(statusLabel(summary.status))}</span></div>
      <div class="summary-item"><span class="summary-label">候选论文数</span><span class="summary-value">${escapeHtml(summary.candidate_count ?? '-')}</span></div>
      <div class="summary-item"><span class="summary-label">入选论文数</span><span class="summary-value">${escapeHtml(summary.selected_count ?? '-')}</span></div>
      <div class="summary-item"><span class="summary-label">Idea 澄清</span><span class="summary-value">${escapeHtml(stageLabel(summary.clarify_stage))}</span></div>
      <div class="summary-item"><span class="summary-label">Query 规划</span><span class="summary-value">${escapeHtml(stageLabel(summary.query_plan_stage))}</span></div>
      <div class="summary-item"><span class="summary-label">检索阶段</span><span class="summary-value">${escapeHtml(stageLabel(summary.retrieval_stage))}</span></div>
      <div class="summary-item"><span class="summary-label">输出目录</span><span class="summary-value">${escapeHtml(summary.output_dir || manifest.output_dir || '-')}</span></div>
      <div class="summary-item"><span class="summary-label">运行日志</span><span class="summary-value">${escapeHtml(runLogPath || '-')}</span></div>
    </div>
    <div class="summary-note">
      <strong>本次任务摘要：</strong>${escapeHtml(summary.idea_excerpt || '-')}
    </div>
    ${keyNotesHtml}
    ${keyWarning ? `<div class="summary-note warning"><strong>关键提示：</strong>${escapeHtml(conciseWarningMessage(keyWarning))}</div>` : ''}
  `;
}

function openCurrentLog() {
  const logPath = currentLoadedRun?.runManifest?.artifacts?.log || (currentRunState?.outdir ? `${currentRunState.outdir}/run.log` : '');
  if (logPath) {
    window.researchFlow.openPath(logPath);
  }
}

refreshRunsBtn.addEventListener('click', refreshRuns);
openRunDirBtn.addEventListener('click', () => {
  if (currentRunDir) {
    window.researchFlow.openPath(currentRunDir);
  }
});
openRunLogBtn.addEventListener('click', openCurrentLog);
openRunLogInlineBtn.addEventListener('click', openCurrentLog);
providerSelect.addEventListener('change', syncProviderOptions);
startRunBtn.addEventListener('click', startRun);
generateClarificationTurnBtn.addEventListener('click', generateClarificationTurn);
appendClarificationAnswerBtn.addEventListener('click', appendClarificationAnswer);
ideaInput.addEventListener('input', () => {
  if (shouldResetClarificationSession(clarificationIdea, ideaInput.value, clarificationHistory.length)) {
    resetClarificationSession('研究想法已修改，旧澄清会话已自动清空。');
  }
});

window.researchFlow.onRunStatus(async (payload) => {
  currentRunState = payload;
  renderCurrentRunStatus(payload);
  if (payload?.status === 'completed' || payload?.status === 'completed_with_warnings') {
    await refreshRuns();
    if (payload.outdir) {
      await loadRun(payload.outdir);
    }
  }
});

loadUiConfig().then(refreshRuns);
