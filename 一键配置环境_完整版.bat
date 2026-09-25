@echo off
rem 切 UTF-8 代码页：本文件含中文提示，Windows 默认 GBK 会显示成乱码。
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   AI Video Agent  -  一键配置环境（完整版）
echo   含 AI 模型依赖：torch + ultralytics + funasr
echo   启用 YOLO 视觉理解 / 语音转写，并跑满 163 测试
echo ============================================
echo.
echo [注意] 完整版会下载 torch/ultralytics/funasr，合计约 2~3GB，耗时较长；
echo        首次运行 M2 还需联网下载 YOLO / Paraformer 权重。仅做演示验收用基础版即可。
echo.

rem --- 1. 检测 Python ---
set PY=
where python >nul 2>&1 && set PY=python
if not defined PY ( where py >nul 2>&1 && set PY=py )
if not defined PY (
  echo [错误] 未检测到 Python。请先安装 Python 3.10+ 并**勾选 "Add Python to PATH"**。
  echo   下载：https://www.python.org/downloads/
  pause
  exit /b 1
)
echo [1/4] Python 就绪：
%PY% --version

rem --- 2. 创建虚拟环境 ---
if not exist ".venv\Scripts\python.exe" (
  echo [2/4] 创建虚拟环境 .venv ...
  %PY% -m venv .venv
  if errorlevel 1 ( echo [错误] 创建虚拟环境失败 & pause & exit /b 1 )
) else (
  echo [2/4] 虚拟环境已存在，跳过创建
)

rem --- 3. 安装完整依赖 ---
echo [3/4] 安装完整依赖（含 torch/ultralytics/funasr，约 2~3GB，耐心等待）...
.venv\Scripts\python.exe -m pip install -r requirements-full.txt
if errorlevel 1 (
  echo [错误] 依赖安装失败，请检查网络后重跑本脚本。
  pause
  exit /b 1
)

rem --- 4. 校验 ffmpeg（缺失则自动下载补齐）与完整依赖 ---
echo [4/4] 校验运行时...
if exist "tools\ffmpeg\bin\ffmpeg.exe" goto ffmpeg_ready
echo   未找到 ffmpeg（源码仓库不含该大二进制），尝试自动下载补齐（约 80MB，需联网）...
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\fetch_ffmpeg.ps1"
if not errorlevel 1 goto ffmpeg_ready
echo   [警告] ffmpeg 自动下载未成功。真实出片需要 ffmpeg，可任选其一：
echo     方式一：使用随附完整交付包 AI视频剪辑智能体-源码.zip（内含 tools\ffmpeg\bin\）
echo     方式二：手动下载 Windows 静态构建，把 ffmpeg.exe / ffprobe.exe 放入 tools\ffmpeg\bin\
goto ffmpeg_done
:ffmpeg_ready
echo   已就位：tools\ffmpeg\bin\ffmpeg.exe（出片工具链）
:ffmpeg_done
.venv\Scripts\python.exe -c "import gradio, numpy, cv2, torch, ultralytics, funasr; print('   完整依赖导入 OK')"

echo.
echo ============================================
echo   配置完成（完整版）！
echo     双击 一键测试.bat        应显示 163 passed（全部跑满）
echo     双击 一键启动界面.bat    界面「② M2 理解」按钮现已可用
echo ============================================
pause
