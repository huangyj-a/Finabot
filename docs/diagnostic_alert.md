# 诊断告警接入指南

`RuntimeService` 内建的 `DiagnosticMonitor` 周期检测运行时异常，命中 `error` 级 issue 时触发告警送达。支持两种送达方式，可同时启用。

---

## 方式一：环境变量（零代码，运维式）

设置 `FINABOT_ALERT_WEBHOOK_URL`，直接填企业微信 / 钉钉 / 飞书 / Slack / 自建监控的 webhook 地址，

```bash
# .env
FINABOT_ALERT_WEBHOOK_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx
```

`RuntimeService` 传 `diagnostic_interval_seconds` 开启诊断（默认 `0` 关闭）：

```python
# cli/commands.py
runtime = RuntimeService(
    agent, bus,
    heartbeat_interval_seconds=30,
    diagnostic_interval_seconds=60,  # 每 60s 检测一次
)
```

告警 payload 示例：

```json
{
  "type": "finabot_diagnostic_alert",
  "ts": "2026-08-29T16:00:00",
  "events": [
    {
      "level": "error",
      "code": "task_error:heartbeat",
      "message": "周期任务 heartbeat 新增错误（累计 3）：akshare api timeout"
    }
  ]
}
```

---

## 方式二：notifier 回调（编程式，灵活）

`DiagnosticMonitor` 接受 `notifier` 参数，同步或异步回调均可：

```python
from finabot.runtime import DiagnosticMonitor, RuntimeService

async def on_alert(errors: list[dict]):
    """errors 是本次检测命中的全部 error 级 issue。"""
    for issue in errors:
        print(f"[ALERT] {issue['code']}: {issue['message']}")
        # 发企业微信 / 钉钉 / 自研告警平台
        await send_to_alert_platform(issue)

# 方式 A：直接构造 DiagnosticMonitor
monitor = DiagnosticMonitor(
    runtime,
    log_dir=Path("memory/runtime"),
    notifier=on_alert,
)

# 方式 B：通过 RuntimeService 注册（需在 start() 前 add_task）
runtime = RuntimeService(agent, bus, diagnostic_interval_seconds=60)
monitor = DiagnosticMonitor(runtime, notifier=on_alert)
runtime.add_task(PeriodicTask("diagnostic", 60, monitor.run, run_immediately=True))
```

---

## 检测规则与级别

| 检测项 | 条件 | 级别 | 送达 |
|--------|------|------|------|
| `heartbeat_stalled` | 两次快照间 `heartbeat_count` 未增长 | warning | ❌ 仅日志 |
| `task_error:<name>` | 周期任务 `error_count` 增量 | error | ✅ 告警 |
| `bus_backlog:inbound` | 入站队列 > warn(10) / error(50) | warn/error | ✅ 仅 error |
| `bus_backlog:outbound` | 出站队列 > warn(10) / error(50) | warn/error | ✅ 仅 error |
| `llm_failure_rate` | 样本量≥5 且失败率>30% | error | ✅ 告警 |
| `llm_high_latency` | 平均耗时>30s | warning | ❌ 仅日志 |

---

## 阈值覆盖

```python
DiagnosticMonitor(runtime, thresholds={
    "bus_backlog_warn": 10,       # 积压 ≥ 10 → warning
    "bus_backlog_error": 50,      # 积压 ≥ 50 → error
    "llm_min_calls": 5,           # 样本量下限
    "llm_failure_rate_warn": 0.3, # 失败率 ≥ 30% → error
    "llm_latency_warn_ms": 30000, # 平均耗时 ≥ 30s → warning
})
```

---

## 诊断日志文件

`diagnostic.log` 位于 `FINABOT_RUNTIME_DIR`（默认 `memory/runtime/`），JSON-lines 格式，每行一条异常记录：

```json
{"seq": 3, "ts": "2026-08-29T16:00:00", "level": "error", "code": "bus_backlog:inbound", "message": "inbound 队列积压 60 条（> 50）"}
```

可直接被 Filebeat、Logstash、Vector 等日志采集器消费。

---

## 经验法则

- 告警送达是**尽力而为**：webhook 不可达、notifier 抛错均不会阻断主流程。
- 进程自身死亡（如 OOM、`SIGKILL`）不在本模块覆盖范围内，需配合 systemd 或 Docker 重启策略。
- 第一条诊断在 `run_immediately=True` 时立即执行，不会等满一个周期才启动。