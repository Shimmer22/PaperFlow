# PaperFlow

PaperFlow 是一个本地优先的论文研究工作流：把一句模糊想法变成结构化检索、候选筛选、单篇简报和最终讨论。

核心链路：

`idea -> 澄清对话 -> idea spec -> query plan -> 多源检索 -> 去重 -> subagent 初筛 -> 主模型 shortlist -> 论文简报 -> 最终讨论 -> 校验`

## 仓库简介（可直接用）

推荐简介（简洁版）：

`本地优先的 AI 论文研究工作流：从研究想法到检索、筛选、深读与结论讨论的一体化闭环。`

推荐简介（技术版）：

`PaperFlow 是一个 API-only 的研究自动化管线，支持多源论文检索、subagent 评估、主模型汇总和 GUI/CLI 双入口。`

## 当前架构状态

- Provider 模式：`openai_compatible_api`（API-only）
- 默认 Provider：`nvidia`
- 可选 Provider：`nvidia`
- 主模型默认开启 thinking，sub 阶段默认关闭 thinking
- GUI 支持多轮澄清会话（含“我不确定，让模型先思考并给出建议”选项）

## 目录结构

```text
research_flow/           核心 Python 包
prompts/                 模型提示词
templates/               Typst 模板
providers/               Provider 配置（GLM / NVIDIA）
scripts/                 启动与维护脚本
ui/                      Electron 前端
outputs/                 每次运行产物
tests/                   单元测试
```

## 安装

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GUI 依赖：

```bash
cd ui
npm install
cd ..
```

## 环境变量

在项目根目录 `.env` 中配置：

```bash
API_KEY=xxx
NV_API_KEY=xxx
```

说明：

- 使用 GLM 时读取 `API_KEY`
- 使用 NVIDIA 时读取 `NV_API_KEY`
- 应用代码本身只读进程环境变量，推荐通过脚本加载 `.env`

## API 申请教程

如果你需要从零申请和配置 API，可参考这个视频教程：

https://www.bilibili.com/video/BV1nnkJBfELS/?vd_source=493919a81a5d477c1315100aac59a0fa

安全建议：

- 不要把真实 `.env` 提交到仓库
- 只提交 `.env.example`
- 本项目已通过 `.gitignore` 默认忽略 `.env` 和运行产物

## Provider 配置

- [glm_api.example.yaml](/Users/hui/Documents/Paper_Flow/providers/glm_api.example.yaml)
- [nvidia.example.yaml](/Users/hui/Documents/Paper_Flow/providers/nvidia.example.yaml)

NVIDIA 当前示例模型列表（可在 GUI 中直接选择）：

- `z-ai/glm4.7`
- `minimaxai/minimax-m2.5`
- `moonshotai/kimi-k2-thinking`
- `moonshotai/kimi-k2-instruct-0905`
- `deepseek-ai/deepseek-v3.2`
- `mistralai/devstral-2-123b-instruct-2512`
- `nvidia/llama-3.3-nemotron-super-49b-v1.5`
- `openai/gpt-oss-120b`
- `openai/gpt-oss-20b`

## 启动方式

### 1) 启动 GUI（推荐）

```bash
./scripts/start_gui.sh
```

### 2) GLM 直跑

```bash
./scripts/run_glm_api.sh --idea "结合波束成形与自注意力"
```

### 3) NVIDIA 直跑

```bash
./scripts/run_nvidia.sh --idea "结合波束成形与自注意力"
```

### 4) 直接 CLI

```bash
python -m research_flow.cli run --help
python -m research_flow.cli clarify-turn --help
```

## 主要命令

- `research-flow run`
- `research-flow clarify-turn`
- `research-flow validate --run-dir outputs/<run_dir>`
- `research-flow provider-check --provider-config providers/glm_api.example.yaml`
- `research-flow ui`

## 产物说明

每次运行会在 `outputs/<run_dir>` 下生成：

- `idea.txt`
- `clarification_history.json` / `clarification_dialogue.json`（如果使用 GUI 澄清）
- `clarified_idea.json`
- `query_plan.json`
- `candidate_papers_raw.json`
- `candidate_papers_merged.json`
- `scout_reports.json`
- `shortlist_decision.json`
- `ranked_candidates.json`
- `selected_papers.json`
- `papers/<paper_id>/paper_brief.json`
- `final_discussion.json` / `final_discussion.md`
- `run_manifest.json`
- `run_summary.json`
- `validation_report.json`
- `run.log`

## 错误与降级策略

当前系统会优先“可运行完成”，而不是中途崩溃：

- 主模型阶段失败（超时/限流/模型不可用）：回退本地规则
- 某检索源失败：记录 warning，继续其余源
- PDF 下载失败：回退摘要级简报
- Typst 编译失败：保留 `.typ` 与编译日志

你会在 `run_summary.json` 的这些字段看到阶段状态：

- `clarify_stage`
- `query_plan_stage`
- `retrieval_stage`

## 常见问题

### 1) NVIDIA 返回 410 Gone

通常是模型下线（EOL），请切换模型（例如 `z-ai/glm4.7` 或 `minimaxai/minimax-m2.5`）。

### 2) 一直 429

- 降低请求频率，间隔重试
- 临时减少检索源（例如只用 `openalex`）
- 避免在短时间内重复点击 GUI 连续运行

### 3) 根目录出现 `research-flow-provider-*`

这是旧临时目录遗留。新版本已修复不再在仓库目录创建。可执行：

```bash
./scripts/cleanup_provider_temp_dirs.sh
```

## 配置入口

- 应用配置：[config.example.yaml](/Users/hui/Documents/Paper_Flow/config.example.yaml)
- Provider 工厂：[factory.py](/Users/hui/Documents/Paper_Flow/research_flow/providers/factory.py)
- API Provider：[api_provider.py](/Users/hui/Documents/Paper_Flow/research_flow/providers/api_provider.py)
- 编排器：[orchestrator.py](/Users/hui/Documents/Paper_Flow/research_flow/orchestrator.py)

## 说明

`providers/codex.example.yaml`、`providers/custom_cli.example.yaml` 目前仅保留为历史示例；当前主流程是 API-only。
