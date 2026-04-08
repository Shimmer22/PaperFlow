const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('researchFlow', {
  listRuns: (outputRoot) => ipcRenderer.invoke('list-runs', outputRoot),
  loadRun: (runDir) => ipcRenderer.invoke('load-run', runDir),
  openPath: (targetPath) => ipcRenderer.invoke('open-path', targetPath),
  getUiConfig: () => ipcRenderer.invoke('get-ui-config'),
  generateClarificationTurn: (options) => ipcRenderer.invoke('generate-clarification-turn', options),
  startRun: (options) => ipcRenderer.invoke('start-run', options),
  getRunState: () => ipcRenderer.invoke('get-run-state'),
  saveRunAlias: (runDir, alias) => ipcRenderer.invoke('save-run-alias', runDir, alias),
  deleteRun: (runDir) => ipcRenderer.invoke('delete-run', runDir),
  onRunLog: (handler) => ipcRenderer.on('run-log', (_, payload) => handler(payload)),
  onRunStatus: (handler) => ipcRenderer.on('run-status', (_, payload) => handler(payload)),
});
