const { app, BrowserWindow, ipcMain, shell } = require('electron');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const YAML = require('yaml');

const workspaceRoot = path.resolve(__dirname, '..', '..');
const providersDir = path.join(workspaceRoot, 'providers');
const defaultConfigPath = path.join(workspaceRoot, 'config.example.yaml');
const defaultOutputRoot = path.join(workspaceRoot, 'outputs');

let currentRunProcess = null;
let currentRunMeta = null;

function createWindow() {
  const win = new BrowserWindow({
    width: 1240,
    height: 860,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadFile(path.join(__dirname, 'index.html'));

  // Keep external links in the user's default system browser.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//i.test(url)) {
      shell.openExternal(url);
      return { action: 'deny' };
    }
    return { action: 'deny' };
  });
  win.webContents.on('will-navigate', (event, url) => {
    const currentUrl = win.webContents.getURL();
    if (url !== currentUrl && /^https?:\/\//i.test(url)) {
      event.preventDefault();
      shell.openExternal(url);
    }
  });
}

function safeReadJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  } catch {
    return null;
  }
}

function safeReadText(filePath) {
  try {
    return fs.readFileSync(filePath, 'utf-8').trim();
  } catch {
    return '';
  }
}

function safeReadYaml(filePath) {
  try {
    return YAML.parse(fs.readFileSync(filePath, 'utf-8'));
  } catch {
    return null;
  }
}

function collectFilesRecursive(rootDir, targetName, files = []) {
  if (!fs.existsSync(rootDir)) {
    return files;
  }
  for (const entry of fs.readdirSync(rootDir, { withFileTypes: true })) {
    const fullPath = path.join(rootDir, entry.name);
    if (entry.isDirectory()) {
      collectFilesRecursive(fullPath, targetName, files);
    } else if (entry.isFile() && entry.name === targetName) {
      files.push(fullPath);
    }
  }
  return files;
}

function detectPythonCommand() {
  const venvPython = path.join(workspaceRoot, '.venv', 'bin', 'python');
  return fs.existsSync(venvPython) ? venvPython : 'python3';
}

function listProviderConfigs() {
  if (!fs.existsSync(providersDir)) {
    return [];
  }
  return fs.readdirSync(providersDir)
    .filter((name) => name.endsWith('.yaml') || name.endsWith('.yml'))
    .map((name) => {
      const filePath = path.join(providersDir, name);
      const config = safeReadYaml(filePath) || {};
      const providerType = config.provider_type || 'cli';
      if (providerType !== 'openai_compatible_api') {
        return null;
      }
      return {
        name: config.name || path.basename(name, path.extname(name)),
        displayName: config.display_name || config.name || name,
        filePath,
        supportedModels: config.supported_models || [],
        supportedReasoningEfforts: config.supported_reasoning_efforts || [],
        apiKeyEnvVar: config.api_key_env_var || '',
        command: config.command || '',
      };
    })
    .filter(Boolean);
}

function sendToAllWindows(channel, payload) {
  for (const win of BrowserWindow.getAllWindows()) {
    win.webContents.send(channel, payload);
  }
}

function buildRunCommand(options) {
  const pythonCmd = detectPythonCommand();
  const args = [
    '-m',
    'research_flow.cli',
    'run',
    '--idea',
    options.idea,
    '--config',
    options.configPath || defaultConfigPath,
    '--provider',
    options.providerName,
    '--provider-config',
    options.providerConfigPath,
    '--candidate-limit',
    String(options.candidateLimit || 5),
    '--max-papers',
    String(options.maxPapers || 5),
    '--sources',
    options.sources || 'openalex,semanticscholar,arxiv',
    '--outdir',
    options.outdir,
  ];
  if (options.downloadPdf) {
    args.push('--download-pdf');
  } else {
    args.push('--no-download-pdf');
  }
  if (options.parallel) {
    args.push('--parallel');
  } else {
    args.push('--no-parallel');
  }
  if (options.mainModel) {
    args.push('--main-model', options.mainModel);
  }
  if (options.mainReasoningEffort) {
    args.push('--main-reasoning-effort', options.mainReasoningEffort);
  }
  if (options.subModel) {
    args.push('--sub-model', options.subModel);
  }
  if (options.subReasoningEffort) {
    args.push('--sub-reasoning-effort', options.subReasoningEffort);
  }
  if (options.clarificationHistoryFile) {
    args.push('--clarification-history-file', options.clarificationHistoryFile);
  }
  return { command: pythonCmd, args };
}

function runCommandCaptureJson(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: workspaceRoot,
      env: process.env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
    child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `command failed: ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout.trim() || '{}'));
      } catch (err) {
        reject(new Error(`invalid json output: ${err.message}`));
      }
    });
  });
}

function buildCompletionMessage(outdir) {
  const summary = safeReadJson(path.join(outdir, 'run_summary.json')) || {};
  const validation = safeReadJson(path.join(outdir, 'validation_report.json')) || {};
  if ((summary.candidate_count || 0) === 0) {
    return '运行完成，但没有检索到候选论文。请优先检查网络、论文源可用性或 query 质量。';
  }
  if ((summary.selected_count || 0) === 0) {
    return '运行完成，但没有论文进入最终 shortlist。';
  }
  if (validation.success === false) {
    return '运行完成，但部分产物不完整，请查看 validation_report.json。';
  }
  return '运行完成。';
}

ipcMain.handle('list-runs', async (_, outputRoot) => {
  const root = path.resolve(outputRoot);
  if (!fs.existsSync(root)) {
    return [];
  }
  return fs.readdirSync(root)
    .map((name) => path.join(root, name))
    .filter((fullPath) => fs.statSync(fullPath).isDirectory())
    .map((runDir) => {
      const summary = safeReadJson(path.join(runDir, 'run_summary.json'));
      const alias = safeReadText(path.join(runDir, 'run_alias.txt'));
      return { runDir, summary, alias };
    })
    .filter((item) => item.summary);
});

ipcMain.handle('load-run', async (_, runDir) => {
  const runManifest = safeReadJson(path.join(runDir, 'run_manifest.json'));
  const runSummary = safeReadJson(path.join(runDir, 'run_summary.json'));
  const candidates = safeReadJson(path.join(runDir, 'ranked_candidates.json'));
  const selected = safeReadJson(path.join(runDir, 'selected_papers.json'));
  const finalDiscussion = fs.existsSync(path.join(runDir, 'final_discussion.md'))
    ? fs.readFileSync(path.join(runDir, 'final_discussion.md'), 'utf-8')
    : '';
  const papersDir = path.join(runDir, 'papers');
  let briefs = [];
  if (fs.existsSync(papersDir)) {
    briefs = collectFilesRecursive(papersDir, 'paper_brief.json')
      .map((file) => safeReadJson(file))
      .filter(Boolean);
  }
  return { runManifest, runSummary, candidates, selected, briefs, finalDiscussion };
});

ipcMain.handle('open-path', async (_, targetPath) => {
  return shell.openPath(targetPath);
});

ipcMain.handle('open-external-url', async (_, targetUrl) => {
  return shell.openExternal(targetUrl);
});

ipcMain.handle('save-run-alias', async (_, runDir, alias) => {
  const aliasPath = path.join(runDir, 'run_alias.txt');
  fs.writeFileSync(aliasPath, String(alias || '').trim(), 'utf-8');
  return { ok: true };
});

ipcMain.handle('delete-run', async (_, runDir) => {
  fs.rmSync(runDir, { recursive: true, force: true });
  return { ok: true };
});

ipcMain.handle('get-ui-config', async () => {
  return {
    workspaceRoot,
    defaultConfigPath,
    defaultOutputRoot,
    pythonCommand: detectPythonCommand(),
    providers: listProviderConfigs(),
    currentRun: currentRunMeta,
  };
});

ipcMain.handle('generate-clarification-turn', async (_, options) => {
  const pythonCmd = detectPythonCommand();
  const args = [
    '-m',
    'research_flow.cli',
    'clarify-turn',
    '--idea',
    options.idea || '',
    '--provider-config',
    options.providerConfigPath,
  ];

  if (options.mainModel) {
    args.push('--main-model', options.mainModel);
  }
  if (options.mainReasoningEffort) {
    args.push('--main-reasoning-effort', options.mainReasoningEffort);
  }

  let historyFile = null;
  if (Array.isArray(options.history) && options.history.length > 0) {
    const tmpDir = path.join(workspaceRoot, '.cache', 'research-flow', 'ui');
    fs.mkdirSync(tmpDir, { recursive: true });
    historyFile = path.join(tmpDir, `clarification_history_${Date.now()}.json`);
    fs.writeFileSync(historyFile, JSON.stringify(options.history, null, 2), 'utf-8');
    args.push('--history-file', historyFile);
  }
  try {
    const payload = await runCommandCaptureJson(pythonCmd, args);
    return { ok: true, payload };
  } catch (error) {
    return { ok: false, error: String(error.message || error) };
  } finally {
    if (historyFile && fs.existsSync(historyFile)) {
      fs.rmSync(historyFile, { force: true });
    }
  }
});

ipcMain.handle('start-run', async (_, options) => {
  if (currentRunProcess) {
    return { ok: false, error: '已有运行中的任务，请等待当前任务结束。' };
  }
  const runOptions = {
    ...options,
    providerName: options.providerName || 'nvidia',
    providerConfigPath: options.providerConfigPath,
    outdir: options.outdir || path.join(defaultOutputRoot, `ui_run_${Date.now()}`),
  };
  let clarificationHistoryFile = null;
  if (Array.isArray(options.clarificationHistory) && options.clarificationHistory.length > 0) {
    fs.mkdirSync(runOptions.outdir, { recursive: true });
    clarificationHistoryFile = path.join(runOptions.outdir, 'clarification_history.json');
    fs.writeFileSync(clarificationHistoryFile, JSON.stringify(options.clarificationHistory, null, 2), 'utf-8');
  }
  runOptions.clarificationHistoryFile = clarificationHistoryFile;
  const built = buildRunCommand(runOptions);
  currentRunMeta = {
    status: 'running',
    outdir: runOptions.outdir,
    providerName: runOptions.providerName,
    mainModel: runOptions.mainModel || '',
    mainReasoningEffort: runOptions.mainReasoningEffort || '',
    subModel: runOptions.subModel || '',
    subReasoningEffort: runOptions.subReasoningEffort || '',
    message: 'CLI 已启动，正在执行研究工作流。',
    command: [built.command, ...built.args],
  };
  currentRunProcess = spawn(built.command, built.args, {
    cwd: workspaceRoot,
    env: process.env,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  currentRunProcess.stdout.on('data', (chunk) => {
    sendToAllWindows('run-log', { stream: 'stdout', text: chunk.toString() });
  });
  currentRunProcess.stderr.on('data', (chunk) => {
    sendToAllWindows('run-log', { stream: 'stderr', text: chunk.toString() });
  });
  currentRunProcess.on('close', (code) => {
    const runSummary = safeReadJson(path.join(runOptions.outdir, 'run_summary.json'));
    currentRunMeta = {
      ...currentRunMeta,
      status: runSummary?.status || (code === 0 ? 'completed' : 'failed'),
      exitCode: code,
      message: code === 0 ? buildCompletionMessage(runOptions.outdir) : '运行失败，请打开 run.log 查看错误。',
    };
    sendToAllWindows('run-status', currentRunMeta);
    currentRunProcess = null;
  });

  sendToAllWindows('run-status', currentRunMeta);
  return { ok: true, run: currentRunMeta };
});

ipcMain.handle('get-run-state', async () => {
  return currentRunMeta;
});

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
