<p align="center">
  <img src="assets/logo.svg" alt="MLEvolve" width="400" />
</p>

# MLEvolve-Alter

MLEvolve-Alter 是 AutoDecision 的算法搜索与代码执行引擎。它以任务说明和数据目录为输入，通过多智能体协作与搜索树持续生成、执行、调试和改进候选方案，最终保存可复用的最优代码、模型或求解器 artifact、指标以及 Top-K 方案。

当前仓库基于 MLEvolve 演进，并面向 AutoDecision 增强了配置化运行、AutoRealize 上下文消费、优化/决策/强化学习流程、任务继续、前端搜索树、token 观测、双层日志和跨平台任务资源限制。

> **许可证状态：待上游确认。** 经检查，上游 MLEvolve 仓库当前未声明许可证。获得明确的修改与再分发授权前，本仓库不能被宣称为开源发行版。源码可见不等于获得使用或再分发许可。

## 适用任务

- 表格回归与分类
- 时序预测与异常检测
- 推荐、排序和其他机器学习任务
- 深度学习、视觉、文本、音频等领域任务
- 数学优化、组合优化和业务决策
- 强化学习、离线 RL、模仿学习和混合策略

MLEvolve 不要求每个任务都使用 RL，也不会把某个业务问题的固定模板硬编码到通用流程中。每个搜索节点可以选择不同的方法，由统一、可执行的评价结果进行比较。

## 工作流程

```text
description.md + AutoRealize context + input data
                    |
                    v
            Phase 1: Draft Generation
       快速首稿 / stepwise 后续草稿 / 灰色 pending 节点
                    |
                    v
            Phase 2: Search & Execution
       execute -> parse -> review -> debug/improve/fuse
                    |
                    v
       best_solution + top_solution + journal + logs
```

### Draft 阶段

- 生成多个相互独立的根方案。
- 第一个 draft 默认使用 `fast_first_draft`，单次生成完整可执行方案，尽快产生首个可见节点。
- 后续 draft 可使用 stepwise 流程，分阶段处理数据、评价器、模型/求解器和训练或搜索逻辑。
- 优化/RL 任务可限制初始 draft 数，将更多预算留给后续 debug 和 improve。
- 尚未完成代码生成或执行的 draft 写入 `pending_nodes.json`，前端可以灰色节点显示。

### 搜索阶段

- 执行候选代码并解析 metric、运行错误、LLM insight 和程序化诊断。
- 对失败节点生成 debug 子节点，对成功节点生成 improve 子节点。
- 根据搜索进度、分支状态和指标选择继续探索或利用高质量方案。
- 支持跨分支 fusion 和全局经验记忆，复用成功经验并避免重复失败。
- 达到步数或时间预算时正常结束，保存当时已有的最佳方案和完整 journal。

### 优化与强化学习

当节点选择优化或 RL 时，stepwise prompt 会引导其建立完整的可执行链路：

- 明确问题对象、决策变量、约束和统一 evaluator。
- 从真实数据推导 state、action、transition、reward、terminal 和候选/action mask。
- 对无合法动作、非法动作、长 horizon 和大规模组合动作给出处理策略。
- 允许自由选择 PPO、DQN、Actor-Critic、offline RL、imitation、hybrid policy 或非 RL 求解器。
- 课程学习、子问题 schedule 和 checkpoint continuation 是可选训练策略，不是固定模板。
- 如果声称使用 RL，最终 rollout、评价和 `predict()` 必须实际使用 policy 或其 artifact，避免只定义未使用的 RL scaffold。

## 主要功能

- MCGS 风格的多分支搜索与渐进式探索/利用。
- Planner、Coder、Reviewer、Result Parser 等多智能体协作。
- 单次完整生成、stepwise 生成和 SEARCH/REPLACE diff patch。
- draft、debug、improve、回溯调试、Top-K 改进和跨分支融合。
- BM25、FAISS、本地或远程 Embedding 的全局记忆。
- 代码执行超时、输出解析、数据泄漏检查和可选 grading server。
- 统一保存节点代码、metric、insight、决策信号和运行结果。
- 最优方案与 Top-K 代码、submission、模型和求解器 artifact 管理。
- 中断后读取原 journal 和工作区继续搜索。
- `MLEvolve.log` 简略日志、`MLEvolve.verbose.log` 详细日志和 LLM token 统计。
- FastAPI 任务服务、前端快照和加速卡清单接口。
- 每任务 CPU、总内存与加速卡可见性配置。

## 功能亮点

### 直接消费 AutoRealize 任务包

当 `data_dir` 指向 AutoRealize 输出目录时，MLEvolve 会在工作区输入中检测 `realize_report/automl_context.md` 或结构化 pack，并将它作为数据访问上下文。`description.md` 继续提供任务目标与评价要求，AutoRealize context 则提供精确 sheet、物理列名、读取方式、约束和输出合同。

生成代码会被提示区分“输入数据列”和“输出结果列”，不能把 submission 字段误当成原始 DataFrame 列。

### 搜索结果可继续利用

只要节点生成了可信、可比较的指标，即使方案仍有改进空间或暂时没有 submission/model artifact，也可以保留在搜索树中供后续 improve/debug，而不是丢弃已有部分成果后从头生成。

### LLM insight 与程序诊断分层

节点同时保存面向人类和后续智能体的 LLM insight，以及 metric、错误、信号等程序 parser 结果。前端优先展示自然语言 insight，后续节点可以同时利用两类信息。

### 配置与运行可观测

搜索预算、draft 策略、重试、模型、思考模式、续写、执行器、日志和文件名都可通过 YAML 控制。运行期持续写入 pending 节点、状态、journal、token 和资源用量文件。

## 环境要求

- Python 3.11 或 3.12，64 位版本
- 可访问一个或多个 OpenAI-compatible LLM API
- 足够的 CPU、内存和磁盘空间执行候选代码
- 可选 GPU/NPU 及其驱动和 Python 运行时

完整 ML/DL 环境可能包含 PyTorch、scikit-learn、XGBoost、LightGBM、优化与 RL 库。建议为 MLEvolve 单独创建虚拟环境，避免领域依赖相互冲突。

> 安全提示：MLEvolve 会执行 LLM 生成的 Python。CPU、内存和设备可见性限制不等于安全沙箱。不要在包含敏感凭据或重要文件的高权限账号下运行不受信任任务，也不要把服务直接暴露公网。

## 安装

```bash
git clone https://github.com/DonaLdZY/MLEvolve-Alter.git
cd MLEvolve-Alter
python -m venv .venv
```

Windows：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 聚合编排运行时与常用数据处理、机器学习、深度学习、数学优化和强化学习依赖。需要额外领域工具时安装：

```bash
python -m pip install -r requirements_domain.txt
```

GPU 用户应先按显卡驱动安装匹配的 PyTorch wheel，再安装其余依赖。外部 `mle-bench` grading server 仅在 `use_grading_server: true` 时需要。

## 配置

[`config/config.yaml`](config/config.yaml) 是唯一的正式默认配置。`run.py` 默认读取该文件，也可以通过 `MLEVOLVE_CONFIG_PATH` 指定任意其他位置的 YAML；命令行点号参数仍可覆盖本次运行配置。

主要配置区：

| 配置 | 作用 |
| --- | --- |
| `data_dir` | 输入数据或 AutoRealize 任务包目录 |
| `desc_file` | `description.md` 等任务说明文件 |
| `log_dir`、`workspace_dir` | 日志与执行工作区根目录 |
| `preprocess_data`、`copy_data` | 输入预处理与复制/链接策略 |
| `resources` | 每任务 CPU、总内存和加速卡可见性 |
| `exec` | 节点执行时限、输出限制和解释器行为 |
| `agent.code` | 代码生成模型、API、thinking、token、重试与续写 |
| `agent.feedback` | 评审/反馈模型与请求策略 |
| `agent.draft` | 快速首稿、后续 stepwise 和 pending 节点 |
| `agent.search` | 步数、时限、并行数、draft/debug/improve 和搜索策略 |
| `agent.memory_*` | 全局记忆与 Embedding 后端 |
| `coldstart` | 可选预训练模型建议 |
| `runtime` | 继续任务、journal、状态与产物文件 |
| `logging` | 简略日志、详细日志、控制台和 LLM usage |

最小示例：

```yaml
data_dir: "/path/to/autorealize-output"
desc_file: "/path/to/autorealize-output/description.md"
exp_id: "demo"
exp_name: "demo"
log_dir: "./runs"
workspace_dir: "./runs"

agent:
  steps: 50
  time_limit: 10800
  code:
    model: "deepseek-v4-pro"
    base_url: "https://api.deepseek.com"
    api_key: ""
    enable_thinking: null
    reasoning_effort: null
    max_tokens: null
  feedback:
    model: "deepseek-v4-pro"
    base_url: "https://api.deepseek.com"
    api_key: ""
    enable_thinking: null
    reasoning_effort: null
    max_tokens: null
  search:
    parallel_search_num: 4
    num_drafts: 8
    num_improves: 5
```

API Key 优先级：

1. YAML 或 CLI 覆盖中的非空 `api_key`
2. `MLEVOLVE_CODE_API_KEY` / `MLEVOLVE_FEEDBACK_API_KEY`
3. `DEEPSEEK_API_KEY`

Embedding 使用 `MLEVOLVE_EMBEDDING_API_KEY`，其次读取 `EMBEDDING_API_KEY`。保存到日志目录的 resolved config 会清除密钥字段。

`max_tokens` 为 `null` 或 `0` 时由 API 服务商决定。`enable_thinking` 和 `reasoning_effort` 会按模型后端映射；后端不支持时不应假定参数一定生效。

## 直接运行

`run.py` 读取 `config/config.yaml`，并使用 OmegaConf 点号参数覆盖配置：

```bash
python run.py --help
```

```bash
python run.py \
  data_dir=/path/to/autorealize-output \
  desc_file=/path/to/autorealize-output/description.md \
  exp_id=demo \
  exp_name=demo \
  log_dir=./runs \
  workspace_dir=./runs \
  agent.steps=50 \
  agent.time_limit=10800 \
  agent.search.parallel_search_num=4
```

PowerShell：

```powershell
python .\run.py `
  data_dir="D:\runs\demo\autorealize" `
  desc_file="D:\runs\demo\autorealize\description.md" `
  exp_id="demo" `
  exp_name="demo" `
  log_dir=".\runs" `
  workspace_dir=".\runs" `
  agent.steps=50 `
  agent.time_limit=10800
```

当 `log_dir` 与 `workspace_dir` 指向同一根目录时，系统会创建：

```text
runs/<timestamp>_<exp-name>/logs/
runs/<timestamp>_<exp-name>/workspace/
```

当两者不同，分别在两个根目录下创建同名运行目录。

## 继续任务

继续任务必须指向原运行的实际日志目录和工作区目录：

```bash
python run.py \
  runtime.resume_run=true \
  log_dir=/path/to/existing/logs \
  workspace_dir=/path/to/existing/workspace \
  data_dir=/path/to/original/data \
  desc_file=/path/to/original/description.md
```

也可以设置：

```bash
MLEVOLVE_RESUME_RUN=1
```

继续时会读取原 `journal.json`、复用已预处理的 `workspace/input`、恢复最佳节点并继续剩余搜索。它不会复原已退出进程占用的内存，也不会恢复某个 Python 进程的瞬时内存状态，因此重新继续后的内存占用通常低于停止前。

## 服务模式

```bash
python -m uvicorn service_api:app --host 127.0.0.1 --port 18103
```

常用接口：

- `GET /health`
- `GET /resources/inventory`
- `POST /jobs/start`
- `GET /jobs/{job_id}`
- `POST /jobs/stop`
- `POST /snapshot`

访问 `http://127.0.0.1:18103/docs` 查看 OpenAPI 文档。AutoDecision 前端通过服务接口提交临时 YAML，而不是直接在 Gateway 进程内运行搜索。

## 每任务资源限制

```yaml
resources:
  cpu_cores: 4
  memory_limit_gb: 8.0
  accelerator_mode: "selected"  # all | selected | none
  accelerator_device_ids: ["cuda:0"]
  monitor_interval_seconds: 0.5
```

- `cpu_cores` 是整个任务进程树共享的逻辑核心预算。
- `memory_limit_gb` 是控制器与全部子进程共享的目标总内存；`0` 表示不限制。
- `accelerator_mode` 控制任务看到全部、指定或不看到加速卡。

平台实现：

- Windows：CPU affinity + Job Object 总内存限制。
- Linux：CPU affinity；优先使用 cgroup v2 `memory.max` 和 `memory.swap.max=0`。
- Linux 无 cgroup 权限：节点 `RLIMIT_AS` 加进程树子进程保护。
- macOS：worker/BLAS 线程预算；节点 `RLIMIT_AS` 加子进程保护。
- CUDA、ROCm、XPU、Ascend 使用对应可见性环境变量。
- Apple MPS 可检测但不可可靠地按进程隐藏。

资源限制的目的，是约束节点和任务进程树，而不是一超限就主动终止整个搜索控制器。实际后端、CPU 编号、峰值与诊断保存在 `resource_usage.json`。

## 输出产物

日志目录常见文件：

```text
logs/
|-- journal.json
|-- filtered_journal.json
|-- run_status.json
|-- pending_nodes.json
|-- config.yaml
|-- best_solution.py
|-- MLEvolve.log
|-- MLEvolve.verbose.log
|-- llm_usage.jsonl
|-- llm_usage_summary.json
|-- llm_usage_brief.json
`-- resource_usage.json
```

工作区常见文件：

```text
workspace/
|-- input/
|-- working/
|-- submission/
|-- best_solution/
|   |-- solution.py
|   |-- metric.txt
|   |-- node_id.txt
|   `-- <model-or-solver-artifacts>
`-- top_solution/
    |-- top1/
    |-- top2/
    `-- ...
```

生成代码应提供可复用入口，例如 `predict(model_path, data)`。启发式或优化求解器也可使用 `model_path=None` 作为统一占位，但仍应保存必要的配置、权重、预处理器或求解器 artifact，并避免在 `predict()` 内重新训练。

## 日志与 token 统计

- `MLEvolve.log`：适合统计阶段、节点、耗时和结果的简略日志。
- `MLEvolve.verbose.log`：包含更完整的调试和库日志。
- `llm_usage.jsonl`：逐调用 token、缓存和模型信息。
- `llm_usage_summary.json`：按模型、阶段和调用类型汇总。
- `llm_usage_brief.json`：供前端和快速成本分析使用的精简汇总。

## 测试

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check agents config engine llm utils run.py service_api.py tests --select E9,F63,F7,F82
```

重点测试覆盖配置与 LLM 参数、AutoRealize context、decision validation、result parser、insight、prompt、任务继续、服务配置和跨平台资源限制。涉及真实模型调用、长时间搜索或 GPU 的测试应单独运行。

## 常见问题

### Draft 阶段很久没有节点

第一个 draft 可能仍在等待 LLM 完整输出、代码 review 或请求重试。查看 `pending_nodes.json`、`MLEvolve.log` 和 `llm_usage.jsonl`。默认 `fast_first_draft` 用于缩短首节点时间，后续草稿才使用较慢的 stepwise。

### 代码读取了不存在的列或 sheet

确认 `data_dir` 指向完整 AutoRealize 输出目录，而不是只复制了 `description.md`。检查输入中是否存在 `realize_report/automl_context.md`，以及日志是否显示使用 AutoRealize context。生成代码还应在读取时对照实际 `df.columns` 和 workbook sheet names 输出诊断。

### 节点有分数但没有 submission 或模型文件

可信的可比较分数可以使节点保留在搜索树中，但日志会记录非致命 warning。后续 improve 节点应补齐 submission 和轻量 artifact，才能形成完整交付方案。

### 为什么继续任务后内存下降

继续任务恢复的是持久化 journal、代码和工作区，不是已终止进程的堆内存、模型实例或缓存。旧 worker 退出后，重新启动只加载当前继续搜索需要的状态。

### 达到时限是否算失败

不是。`steps_completed` 和 `time_limit_exhausted` 都是正常搜索终止原因，系统应保存现有最佳结果并允许生成报告或后续继续。

## 上游项目与致谢

MLEvolve-Alter 基于 MLEvolve、AutoMLGen 及相关 agentic machine-learning engineering 工作演进。感谢 AIDE、InternAgent 和开源机器学习社区提供的研究与工程基础。

- MLEvolve Project Page: <https://internscience.github.io/MLEvolve/>
- AutoMLGen: <https://arxiv.org/abs/2510.08511>
- InternAgent 1.5: <https://arxiv.org/abs/2602.08990>

使用或发布本仓库前，请同时检查仓库当前许可证和上游项目的引用要求。
