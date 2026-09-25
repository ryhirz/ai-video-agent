@echo off
rem 切 UTF-8 代码页：本文件含中文提示，Windows 默认 GBK 会显示成乱码。
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   AI Video Agent  -  Gradio Demo UI
echo   Open: http://127.0.0.1:7860
echo   (若 7860 被旧实例占用，会自动改用邻近端口 —— 以本窗口打印的实际地址为准)
echo   Close this window to stop the server.
echo ============================================
echo.

rem --- Stop stale UI instances left over from previous runs ---
rem --- Only matches this project's UI process, other python services are untouched ---
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*ai_video_agent.ui*' } | ForEach-Object { Write-Host ('  [cleanup] stopping stale UI PID ' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
timeout /t 1 /nobreak >nul

echo.
".venv\Scripts\python.exe" -m ai_video_agent.ui
pause
