@echo off
rem 切 UTF-8 代码页：本文件含中文提示，Windows 默认 GBK 会显示成乱码。
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   AI Video Agent  -  Full Test Suite
echo ============================================
echo.
".venv\Scripts\python.exe" -m pytest tests -q
echo.
echo --------------------------------------------
rem 判定用**退出码**，不写死用例条数 —— 之前写死"Expect: 40 passed"，
rem 用例涨到 163 后这个数字就成了误导（用户会以为跑少了或跑错了）。
if errorlevel 1 (
  echo   FAILED - 见上方失败项
) else (
  echo   ALL PASSED - 用例数见上方统计行
)
echo --------------------------------------------
pause
