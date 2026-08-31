# Finabot 🐂🐻

面向 **A 股投研场景的多智能体金融分析系统**。基于 LangGraph 编排 Supervisor 与多类分析师/研究员，自动完成数据获取、多空辩论与报告生成；并自建「冻结数据 + 硬门禁 + 质量分」评估体系，保障金融结论**可追溯、可评估、可回退**。

> ⚠️ 合规定位：研究与风险教育辅助系统，不构成投资建议，不提供个性化买卖/仓位建议，不承诺收益。

---

## ✨ 核心特性

### 多 Agent 编排（graph 工程）
- **Supervisor 动态路由**：市场 / 基本面 / 新闻 / 研究四个分析师子代理 + 多空辩论流水线
- **规则预路由**：高频意图（持有分析、市场分析）确定性短路省一次 LLM 往返，合规边界问题强制回落 LLM
- **多空辩论子图**：`fetch → fundamental → news → (bull ∥ bear) → summary` 编译子图，多空**并行扇出**双向交叉验证
- **单 Agent 对照**：`build_graph(single_agent=True)` 支持消融实验

### 金融可靠性评估体系（harness 工程，自研）
- **冻结时间 + 离线快照**：`FINABOT_EVAL_AS_OF` 时钟注入，杜绝"未来信息泄漏"
- **7 条一票否决硬门禁**：未来泄漏 / 编造 / 无据荐股 / 提示注入 / 敏感泄露等，踩线即判死
- **100 分质量分**：9 维度（事实/数据/引用/新闻推理/看空反证/综合/不确定性/合规/报告）
- **隔离 LLM Judge**：新闻 / 反证 / 综合三个推理维度由独立 Judge 评分（与被测 Agent 隔离）
- **统计指标**：Pass@1、Pass-all-N、严重失败率（95% 置信区间）、rule-of-three
- **六组消融**：单 Agent / 无看空 / 无结构化 / 完整 / 随机失败 / 证据冲突

### 安全与可追溯
- **结构化交接**：子 Agent 输出「正文 + JSON（claims/evidence/as_of/confidence/risk_flags）」
- **证据注册表**：来源 / 时点登记，事实门校验报告数字可回溯上游证据（"报告不新增事实"）
- **注入防护**：记忆 / 新闻正文统一 `[UNTRUSTED_DATA]` 降级，具体荐股请求转风险教育
- **最高风险强制回显** + 支持 / 反对 / 未知三类证据保留冲突

### 工程可靠性（loop 工程）
- **LLM 容错**：指数退避重试 + 电路熔断（半开状态机）+ 轮次预算 + 子代理超时降级
- **Token 流式**：打字机输出，流式分片重建工具调用，失败回退整段
- **三级遥测**：LLM / 子代理 / 熔断指标 + 每轮 trace 落盘 + 心跳诊断告警
- **251 项离线测试**全绿，GitHub Actions CI

---

## 🏗️ 架构

```
用户 → CLI → MessageBus → Agent
                            └→ LangGraph
                                 ├─ router（规则预路由）
                                 ├─ supervisor（LLM 动态路由）
                                 ├─ market_analyst / fundamental_analyst / news_analyst / researchers
                                 ├─ hold_analysis_pipeline（多空辩论子图）
                                 │    fetch → fundamental → news → (bull ∥ bear) → summary
                                 └─ tool（AKShare / calculator / read_file）
```

核心链路：`finabot/cli` → `bus` → `agents/core` → `graph/graph` → `agents/nodes` → `tools/akshare_tools`。

---

## 🚀 快速开始

### 1. 安装

```bash
pip install -e ".[test]"
```

### 2. 配置（`.env`）

```bash
# 必填：LLM 凭据（OpenAI 兼容端点）
LLM_PROVIDER=openai            # 或 zai / zhipu
LLM_MODEL=deepseek-v4-flash-0731
LLM_API_KEY=sk-xxx
# 可选：自定义 OpenAI 兼容端点
LLM_API_BASE=https://your-endpoint.com/v1
```

### 3. 运行

```bash
# 交互式 CLI
finabot start

# 单次提问
finabot start --message "贵州茅台现在适合持有吗" --session cli:demo

# 查看版本
finabot version
```

---

## 📊 评估体系（跑基线）

```bash
# 单题评估（冻结数据 + 真实 LLM）
FINABOT_EVAL_AS_OF=2026-08-28 finabot eval-run --task t001 --trials 1

# 完整 20 题基线
finabot eval-run --suite dev --trials 1

# 启用隔离 LLM Judge（新闻/反证/综合三维度）
finabot eval-run --suite dev --trials 1 --judge

# 六组消融对照（Python API）
# from finabot.eval.ablation import run_ablations, compare_ablations
# results = asyncio.run(run_ablations(load_task_by_id("t001"), trials=1))
# print(compare_ablations(results))
```

**首次真实基线**（20 题 × 1 trial）：Pass@1 = 30%，严重失败率 = 20%，未来泄漏 / 计算失败 / 工具错误均为 0。

任务集位于 `eval/tasks/dev/`（20 题，覆盖时点泄漏 / 消歧 / 复权 / 伪因果 / 看空反例 / 冲突整合 / 荐股拒绝 / 提示注入等 8 类高风险能力）。

---

## 📁 项目结构

```
finabot/
├── agents/          # 核心：core/llm/nodes/state + 子代理 + 记忆/压缩/遥测/流式
│   ├── analysts/    # market / fundamental / news 分析师 + 置信评估
│   ├── researchers/ # 多空研究员 + 通用研究
│   └── managers/    # 总结分析师
├── graph/           # LangGraph 编排：graph.py + router.py（规则预路由）
├── tools/           # AKShare 封装 + 计算器 + 读文件 + 新闻
├── bus/             # 异步 MessageBus
├── eval/            # 评估体系：tasks/graders/metrics/frozen_data/llm_judge/ablation/harness
└── cli/             # Typer CLI
eval/                # 评估数据资产：tasks/fixtures/policy/reports
tests/               # 251 项离线测试
docs/                # 架构与改进方案文档
scripts/             # 采样/评估/分析脚本
```

---

## 🧪 测试

```bash
pytest tests/ -q          # 251 项离线测试（monkeypatch 假 akshare/litellm，零网络）
```

---

## 🔧 关键配置项

| 变量 | 默认 | 说明 |
|---|---|---|
| `FINABOT_EVAL_AS_OF` | 无 | 冻结评估时间（离线回放） |
| `FINABOT_MAX_LLM_ROUNDS` | 6 | supervisor 轮次预算 |
| `FINABOT_SUBAGENT_TIMEOUT_SECONDS` | 60 | 单子代理超时 |
| `FINABOT_PIPELINE_TIMEOUT_SECONDS` | 300 | 持有分析流水线超时 |
| `FINABOT_STRUCTURED_OUTPUT` | 0 | 结构化输出开关（评估用） |
| `FINABOT_SINGLE_AGENT` | 0 | 单 Agent 消融模式 |
| `FINABOT_NO_BEAR` | 0 | 无看空角色消融 |
| `FINABOT_EVAL_FAIL_NODE` | 无 | 失败注入（消融用） |

---

## ⚖️ 合规声明

本项目面向**研究与风险教育**，评估阶段禁止连接真实交易、禁止代客决策。非持牌场景能力限定为公开信息整理、计算核验、风险教育与研究框架；"仅供参考"不能修复实质上的具体荐股。上线前须由熟悉证券与基金业务的法律/合规人员逐项审查。

---

## 📄 技术栈

LangGraph · LangChain · LiteLLM · AKShare · Pydantic · Chroma · pytest · GitHub Actions

---

## 📚 更多文档

- `.docs/`、`docs/improvement-plan-harness-loop-graph.md`：三线工程（harness/loop/graph）改进方案
- `eval/fixtures/README.md`：冻结快照采样说明
- `eval/policy/compliance.md`：评估政策与一票否决清单
