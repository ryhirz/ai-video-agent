# AI 视频剪辑智能体 — 全栈层 API 接口设计文档

> 版本：v1.0 · 日期：2026-10-23
> 上游依据：PRD v1.0 · 现有实现 `server/app.py`（5 路由 + /manifest + /health，与代码逐条核对）
> Base URL：`http://localhost:8000`（`uvicorn server.app:app` 默认）
> 认证（可选，2026-10-23 落地）：环境变量 `API_KEY` **未设置 = 无认证**（历史行为）；
> **设置后** `/render*` 四条业务路由要求 `X-API-Key` 请求头**或** `?api_key=` 查询参数
> （等价放行，后者服务 video.src / `<a download>` 场景），不匹配 → 401。
> `/manifest` 与 `/app` 静态页保持公开。user_id 默认 "local"（多租户不在范围）

---

## 1. 接口总览

| 方法 | 路径 | 状态码 | 说明 | 实现状态 |
|---|---|---|---|---|
| POST | /render | 202 | 提交渲染任务 | ✅ 第一步 |
| GET | /render/{task_id} | 200 | 状态/进度查询 | ✅ 第一步 |
| POST | /render/{task_id}/cancel | 200 | 软取消 | ✅ 第一步（第二步改 DB 查询态） |
| GET | /render/{task_id}/download | 200 | 成片下载 | ✅ 第一步 |
| GET | /manifest | 200 | 能力清单 | ✅ 第一步 |
| GET | /health | 200 | 健康探针（公开） | ✅ 可观测性起步 |
| POST | （webhook 回调，服务端→用户） | — | 任务终态回调 | ⬜ 第四步 |

## 2. POST /render — 提交渲染任务

**请求体**：

```json
{
  "timeline": { "schemaVersion": "1.0.0", "...": "见交付文档-全栈层.md 第十节" },
  "options": {},
  "user_id": "local",
  "webhook_url": "https://example.com/callback"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| timeline | object | ✓ | 时间线 JSON；合法性由编译器 `compile` 预检异步校验 |
| options | object | — | 渲染选项，默认 `{}` |
| user_id | string | — | 默认 `"local"`（无认证版） |
| webhook_url | string | — | ✅ 第四步起生效：任务终态后服务端 POST 此地址（见 §7） |

**成功 202**：

```json
{ "task_id": "3f9c2a10-...", "state": "pending" }
```

**错误**：422（Pydantic 校验失败：缺 timeline / 类型不符）

## 3. GET /render/{task_id} — 状态查询

**成功 200**：

```json
{
  "task_id": "3f9c2a10-...",
  "state": "running",           // pending|running|success|failed|cancelled|timeout
  "progress": 0.42,
  "result_url": "renders/3f9c...mp4",   // 仅 success 非空
  "error_msg": null             // failed/timeout 时为可读原因
}
```

**错误**：404 `{"detail": "Task not found"}`

> 第三步已实现：progress 由 ffmpeg `time=` 流式解析驱动（≥1s 节流写库，0.0→1.0 连续）。

## 4. POST /render/{task_id}/cancel — 软取消

**语义（拍板：DB 查询态）**：
- 仅 `pending`/`running` 态生效：置 `state=CANCELLED` + `finished_at`；worker 在四处检查点（出队 / render 前 / render 后 / 入库前）查 DB 发现 CANCELLED 即停止
- 对终态任务**幂等**：不修改任何字段，直接返回当前状态
- 已实现（第二步）：API 只写 DB，不接触 worker；executor 内存标志位已删除

**请求/成功 200**：

```json
{ "task_id": "3f9c2a10-...", "state": "cancelled" }
```

**错误**：404（任务不存在）

## 5. GET /render/{task_id}/download — 成片下载

**成功 200**：`video/mp4` 文件流（`FileResponse`，filename = result_url）

| 状态码 | 条件 | detail |
|---|---|---|
| 400 | 任务非 SUCCESS 态或 result_url 为空 | `Task not completed` |
| 404 | 任务不存在 | `Task not found` |
| 410 | 记录在但文件已被清理 | `Result file expired` |

## 6. GET /manifest — 能力清单

**成功 200**：返回 `ai_video_agent/registry::capability_manifest()` 全量 JSON（`capability_manifest.json` 同源，registry 单一真相）。

用途：前端动态表单数据源；新增操作只改 registry，接口与前端零改动。

## 6.1 GET /health — 健康探针（✅ 可观测性起步）

**成功 200**：

```json
{ "status": "ok", "version": "0.2.0" }
```

- **恒公开**：容器编排（docker-compose / K8s）探针无法携带 API Key，
  `API_KEY` 开启时本端点也不要求凭证（与 /manifest、/app 同属公开面）
- 语义：进程可达性（liveness）——不探 DB/broker 深度依赖；就绪细节
  留待后续按需扩展 /ready
- 用途：compose `healthcheck` / K8s `livenessProbe`/`readinessProbe` /
  负载均衡器健康检查的直接对接点

## 7. Webhook 回调契约（✅ 第四步已实现，契约不变）

- 触发时机：任务进入终态（success / failed / cancelled / timeout）后回调
- 方向：服务端 POST 用户提交的 `webhook_url`
- 请求体（与 GET /render/{id} 响应同构 + 终态时间戳）：

```json
{
  "event": "task.finished",
  "task_id": "3f9c2a10-...",
  "state": "success",
  "progress": 1.0,
  "result_url": "renders/3f9c...mp4",
  "error_msg": null,
  "finished_at": "2026-10-23T02:00:00+00:00"
}
```

- 重试策略：非 2xx 视为失败，最多重试 3 次，间隔 10s/60s/300s（指数退避简化版）
- 超时：单次回调 10s
- 安全：本版不加密签（无认证体系的一部分），文档明示调用方应自行校验 task_id

## 8. 错误模型总表

| 状态码 | 场景 |
|---|---|
| 404 | task_id 不存在（查询/取消/下载通用） |
| 400 | 下载但任务未完成 |
| 410 | 成片文件已过期清理 |
| 422 | 请求体 Pydantic 校验失败 |
| 500 | 未预期异常（FastAPI 默认），不应出现——任务级错误一律落 FAILED 状态而非 5xx |

## 9. API 端到端测试（现有 5 条，回归护栏）

`tests/test_server_api.py`：提交→轮询→下载全链路、404、非 SUCCESS 下载 400、取消幂等、manifest 与 json 同源。第二步迁移（Celery eager 模式）必须保持这 5 条不变通过。
