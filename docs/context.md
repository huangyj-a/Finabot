# 上下文压缩 — 预处理管线 + 自动压缩 + 应急兜底
---

## 设计原则
- 便宜的先跑，贵的后跑
- 能改文本 → 不删整条
- 能删整条 → 不调 LLM

## 代价递增
文本操作 → LLM 摘要 → 应急裁剪
`0 API` → `0 API` → `0 API` → `1 API` → `1 API`

---

## 预处理管线（执行顺序：L3 → L1 → L2，每轮 LLM 调用前自动执行，0 API）
| 层级 | 名称 | 规则 | 触发条件 | 效果 |
| :--- | :--- | :--- | :--- | :--- |
| L3 | `toolResultBudget` | `tool_result` 总和 > 200KB → 最大项落盘 | 每轮自动，必须在 `microCompact` 之前保留完整内容 | 保留完整内容 |
| L1 | `snipCompact` | 消息 > 50 条 → 裁掉中间 | 消息数超过阈值 | 保留头尾 |
| L2 | `microCompact` | 旧 `tool_result` → 占位符（保留最近 3 条） | 每轮自动，教学版用文本占位符模拟 | 压旧结果 |

---

## 自动压缩决策（预处理不够时触发，1 API 调用）
| 层级 | 名称 | 规则 | 阈值/熔断 | 调用成本 |
| :--- | :--- | :--- | :--- | :--- |
| L4 | `autoCompact` | token 超阈值 → LLM 全量摘要 | 阈值：`contextWindow - maxOutputTokens - 13,000`<br>先尝试 `sessionMemoryCompact`，不够才调 LLM<br>熔断：连续失败 3 次后停止重试 | 1 API 调用 |

---

## 应急兜底（API 仍然返回 `prompt_too_long` 时触发）
| 层级 | 名称 | 规则 | 效果 |
| :--- | :--- | :--- | :--- |
| 应急 | `reactiveCompact` | API 返回 413 / `prompt_too_long` → 字节级裁剪<br>保留最后 5 条 + 摘要，比 `autoCompact` 更激进 | 应急兜底，强行把 prompt 压到可发送范围 |
