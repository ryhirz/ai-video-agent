"""全栈层（第一步骨架）：FastAPI + SQLite + 本地 FS。

设计对应《全栈层任务队列架构》第一步：
- 执行器用 ThreadPoolExecutor(max_workers=1) 占位，接口形态与未来 Celery 版一致；
- 编译/渲染复用现有 compiler.compile() / render.render() 纯函数，编译器零改动；
- 存储抽象成接口，开发用本地 FS，生产切 MinIO/S3 时只换 storage 实现。
"""
