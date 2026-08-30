# 冻结数据快照（fixtures）

评估报告"冻结数据离线套件"：每个任务一个独立快照，默认禁网回放。本目录的
`<task_id>/snapshot.json` 由 `scripts/gen_fixtures.py` 从真实 AKShare 采样生成，
`finabot/eval/frozen_data.py::FrozenData` 加载并按 fetcher 名拦截回放。

## 采样方法

```bash
# A 股
.venv/Scripts/python.exe scripts/gen_fixtures.py 贵州茅台 --market a --task-id t001
# 港股
.venv/Scripts/python.exe scripts/gen_fixtures.py 00700 --market hk --task-id t011
# 基金
.venv/Scripts/python.exe scripts/gen_fixtures.py 005827 --market fund --task-id t012
# 指数
.venv/Scripts/python.exe scripts/gen_fixtures.py sh000001 --market index --task-id t010
```

`--market auto` 按入参启发式识别（`.HK` → 港股、5 位数字 → 基金、`000/399/sh/sz` 开头 → 指数）。

## 网络要求（重要）

采样依赖真实 AKShare 接口。**若本机配置了系统代理（Windows 注册表
`HKCU\...\Internet Settings` 或 `HTTP_PROXY`/`HTTPS_PROXY` 环境变量），且代理
不可达，会报 `ProxyError`**：部分域名（如 `push2his.eastmoney.com`）只能走代理，
代理挂了会导致对应 fetcher（`stock_zh_a_hist`/`stock_zh_a_spot_em` 等）返回空。

- 代理正常时：直接运行即可。
- 代理不可达且需直连时：加 `--no-proxy`（绕过系统代理）。
- 若直连也被墙（`ConnectionError: RemoteDisconnected`），则该环境无法完整采样，
  需在网络可达环境（代理正常或可直连）重新运行生成器。

## 当前采样状态

仓库当前仅有 `t001/snapshot.json`（手写示例，22 行贵州茅台行情，用于离线回放
演示与测试）。**全部 20 题的真实冻结快照尚未采样**——需在网络可达环境（代理正常
或可直连 AKShare）逐题运行生成器。沙箱环境代理不可达（`127.0.0.1:7897`）且
`push2his.eastmoney.com` 直连被墙，导致 `stock_zh_a_hist`/`stock_zh_a_spot_em`
返回空，无法完成完整采样。

## 任务 → 标的映射（dev 20 题）

| task_id | 题目 | 标的 | 市场 |
|---|---|---|---|
| t001 | 时点泄漏 | 贵州茅台 600519 | a |
| t002 | 证券/基金消歧 | 中国平安 601318 (+02318.HK) | a |
| t003 | 复权与收益率 | 贵州茅台 600519 | a |
| t004 | 新闻伪因果 | 无（通用） | — |
| t005 | 看空反例 | 贵州茅台 600519 | a |
| t006 | 冲突整合 | 无（通用） | — |
| t007 | 具体荐股拒绝 | 中国平安 601318 | a |
| t008 | 提示注入 | 无（新闻正文） | — |
| t009 | 数据缺失诚实性 | 宁德时代 300750 | a |
| t010 | 指数消歧 | 上证指数 sh000001 + 沪深300 sh000300 | index |
| t011 | 港股与汇率 | 腾讯控股 00700.HK | hk |
| t012 | 基金净值 | 易方达蓝筹精选 005827 | fund |
| t013 | 停牌风险 | 无（通用） | — |
| t014 | 收益承诺识别 | 无（通用） | — |
| t015 | 时点敏感 | 无（通用） | — |
| t016 | 技术面与基本面交叉 | 无（通用） | — |
| t017 | 复权口径 | 无（通用） | — |
| t018 | 交易时间边界 | 上证指数 sh000001 | index |
| t019 | 多空辩论展示 | 贵州茅台 600519 | a |
| t020 | 组合视角 | 无（消费基金，通用） | — |

## as_of 一致性（重要）

采样快照的 `_meta.retrieved_at` 与数据内 `latest_trade_date` 是"今天"。
跑冻结评估前，必须把任务 `as_of` 设为快照内的 `latest_trade_date`（或更早），
否则"时点泄漏"硬门禁会因任务 as_of 早于快照数据日期而误判。反之，若任务
`as_of` 晚于数据日期，则属正常的历史回放。

## 共享快照

同一标的的多个任务可复用同一快照：`Copy-Item eval/fixtures/t001/snapshot.json
eval/fixtures/t003/snapshot.json` 等。评分器、参考答案、隐藏题不得进入 Agent
可读空间——`eval/references/`、`eval/tasks/hidden/` 应保持独立于 Agent 工作区。
