# AI 视频剪辑智能体 — 全栈层 PRD 需求文档

> 版本：v1.0 · 日期：2026-10-23
> 上游依据：《交付文档-全栈层.md》（里程碑/验证基线）· 核心库能力：`capability_manifest`（registry 单一真相）
> 权威优先级：本文档（PRD）> UI/UX > API 设计 > 数据库设计 > 架构说明

---

## 1. 产品定位

自然语言驱动的视频剪辑智能体的 **Web 服务化层**：将核心库（编译器 + ffmpeg 渲染）封装为异步渲染服务，提供任务提交、进度查询、软取消、成片下载与能力清单查询，支撑前端预览页与时间轴编辑器。

**核心价值**：前端不直接依赖 Python 核心库，通过 HTTP API + 能力清单（manifest）实现「前端动态表单 ← registry 自动跟随」的解耦。

**目标用户**：
- 剪辑用户（通过前端页面提交渲染任务）
- 前端开发者（消费 /manifest 与渲染 API）
- 运维/评审者（通过状态机与错误信息定位问题）

## 2. 功能清单

| 编号 | 功能 | 优先级 | 里程碑 | 说明 |
|---|---|---|---|---|
| F1 | 提交渲染任务（timeline JSON） | P0 | 第一步 ✅ | POST /render → 202 + task_id |
| F2 | 任务状态/进度查询 | P0 | 第一步 ✅ | progress 0.0→1.0 轮询 |
| F3 | 成片下载 | P0 | 第一步 ✅ | 仅 SUCCESS 态可下载 |
| F4 | 能力清单查询 | P0 | 第一步 ✅ | 前端动态表单数据源 |
| F5 | Celery + Redis 异步执行 | P0 | **第二步** | `render_video.delay` 替换 ThreadPool 占位 |
| F6 | 软取消（DB 查询态） | P0 | **第二步** | 拍板：API 只置 DB state=CANCELLED，worker 三个检查点查 DB |
| F7 | 测试策略落地（eager 为主） | P0 | **第二步** | 拍板：`task_always_eager=True` 为主 + 真队列冒烟 skipif（本机暂无 Redis，后补） |
| F8 | 依赖显式化 | P0 | **第二步** | requirements.in 显式声明 celery[redis]/fastapi/uvicorn/sqlalchemy |
| F9 | ffmpeg 进度解析（`time=` → progress） | P1 | 第三步 | 流式解析替代检查点间 0.0/1.0 二值进度 |
| F10 | 流式取消（两步杀进程） | P1 | 第三步 | `proc.terminate` → 宽限 → `proc.kill` |
| F11 | TIMEOUT 启用 | P1 | 第三步 | `soft_time_limit=540` 捕获后优雅落 TIMEOUT；`time_limit=600` 硬杀兜底 |
| F12 | webhook 回调 | P2 | 第四步 | 字段已入库未实现，回调契约见 API 文档 §7 |
| F13 | MinIO 对象存储 | P2 | 第四步 | Storage 接口已抽象，切实现零改动上层 |
| F14 | 最小预览页 | P1 | 前端·阶段1 | 提交/查询/下载三区单页 |
| F15 | 时间轴编辑器 | P2 | 前端·阶段2 | 轨道/属性面板/manifest 驱动表单 |

## 3. 用户故事与验收标准

### US-1 提交渲染任务
> 作为剪辑用户，我想提交 timeline JSON 并获得成片下载链接，以便无需接触 Python 环境完成剪辑。

**验收标准**：
- [ ] POST /render 携带合法 timeline 返回 202 与 `{"task_id", "state": "pending"}`
- [ ] timeline 非法（schema 不符/操作不支持）任务落 FAILED，`error_msg` 含可读原因
- [ ] 每次提交生成全局唯一 UUID task_id

### US-2 查询进度
> 作为用户，我想随时查看渲染进度，以便知道还要等多久。

**验收标准**：
- [ ] GET /render/{id} 返回 state/progress/result_url/error_msg
- [ ] 不存在的 task_id 返回 404
- [ ] progress 单调不减，终态为 1.0（SUCCESS）或保持最后值

### US-3 取消任务
> 作为用户，我想取消排队中或执行中的任务，以便不再浪费等待时间。

**验收标准**（拍板：DB 查询态）：
- [ ] 对 PENDING/RUNNING 任务 POST cancel → 状态置 CANCELLED 并返回
- [ ] worker 在出队 / render 前 / render 后 / 入库前**四处检查点**查 DB，发现 CANCELLED 即停止（出队检查必须先于 RUNNING 写入，否则取消信号被抹掉）
- [ ] 对 SUCCESS/FAILED/CANCELLED/TIMEOUT 任务重复 cancel **幂等**：直接返回当前状态，不改库
- [ ] 不存在的 task_id 返回 404

### US-4 动态表单
> 作为前端开发者，我想通过 /manifest 获取全部支持的操作与参数，以便不改前端代码即可跟随核心库新增能力。

**验收标准**：
- [ ] GET /manifest 返回与 `capability_manifest.json` 一致的能力清单
- [ ] 核心库新增操作后，manifest 自动包含（registry 单一真相，无手工同步）

### US-5 异步与稳定
> 作为运维，任务超时/失败要有明确状态与错误信息，以便统计与排障。

**验收标准**：
- [ ] 任务体捕获 `SoftTimeLimitExceeded`（540s）→ 优雅落 TIMEOUT + error_msg
- [ ] `time_limit=600` 硬杀兜底：`acks_late=True + reject_on_worker_lost=True` 下任务重回队列，任务体入口检查「created_at 距今已超限」直接落 TIMEOUT，防死循环
- [ ] 任何未捕获异常落 FAILED + error_msg，不留 RUNNING 僵尸
- [ ] 全量回归：`pytest tests -q` 通过且文档数字与实际一致

## 4. 非功能需求

| 类别 | 要求 |
|---|---|
| 性能 | 单机 1 worker（`--concurrency=1`，prefork）；API 响应 < 100ms（不含渲染） |
| 可靠性 | 状态机六态合法流转，禁止越态；临时文件用后即清 + TTL 清理 |
| 可测试性 | eager 为主（CI 零外部依赖）；真队列冒烟测试 `skipif` 无 Redis 自动跳过 |
| 安全 | 无硬编码密钥；参数绑定（DAL 已参数化）；timeline 为用户输入，仅进编译器校验链路，不拼入 shell |
| 兼容 | 编译器核心库零改动；SQLite→SQLAlchemy 表结构字段名不变 |

## 5. 里程碑与范围边界

| 里程碑 | 范围 | 明确不做 |
|---|---|---|
| 第二步 | F5–F8 | 不做流式取消、webhook、MinIO |
| 第三步 | F9–F11 | 不做对象存储切换、回调 |
| 第四步 | F12–F13 | 不做多用户/认证体系 |
| 前端阶段1 | F14 | 不做时间轴编辑 |
| 前端阶段2 | F15 | — |

**已知边界**：当前 API 无认证（user_id 默认 "local"），多租户与鉴权不在本版范围，留待后续迭代。
