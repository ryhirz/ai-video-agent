# ============================================================
# AI 视频剪辑智能体 —— 生产镜像
# 基础：python:3.13-slim + apt ffmpeg（resolve_ffmpeg 依次回退：
#       FFMPEG_BIN → tools/ffmpeg/bin → PATH，容器内走 PATH）
# 依赖：requirements.txt（pip-compile 哈希锁，--require-hashes 安装）
# 健康检查：GET /health（第十七节交付的探针端点，恒公开）
# ============================================================
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ffmpeg 走 apt（仓库内 tools/ffmpeg 是 Windows 二进制，Linux 容器不适用，
# 由 .dockerignore 排除；resolve_ffmpeg 末尾回退 shutil.which 即命中）
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖再拷代码——依赖层缓存最大化，改代码不重装
COPY requirements.txt .
RUN pip install --no-cache-dir --require-hashes -r requirements.txt

COPY server/ server/
COPY ai_video_agent/ ai_video_agent/

# /health 探针（无 curl 的 slim 镜像用 python 标准库探活）
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
