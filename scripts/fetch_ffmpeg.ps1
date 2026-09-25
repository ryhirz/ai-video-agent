<#
  fetch_ffmpeg.ps1
  ---------------------------------------------------------------
  自动下载 Windows 版 ffmpeg 静态构建（ffmpeg.exe / ffprobe.exe），
  放入 tools\ffmpeg\bin\，供「从 GitHub 克隆源码」后补齐出片工具链。

  背景：ffmpeg 三个 exe 各约 156MB，超过 GitHub 单文件 100MB 硬限制，
        因此源码仓库不包含它们；本脚本在缺失时自动补齐。

  用法（由 一键配置环境.bat 自动调用）：
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fetch_ffmpeg.ps1
  ---------------------------------------------------------------
#>
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$root    = Split-Path -Parent $PSScriptRoot
$binDir  = Join-Path $root 'tools\ffmpeg\bin'
$ffmpeg  = Join-Path $binDir 'ffmpeg.exe'
$ffprobe = Join-Path $binDir 'ffprobe.exe'

if ((Test-Path $ffmpeg) -and (Test-Path $ffprobe)) {
  Write-Host "[skip] ffmpeg 已存在，无需下载：$binDir"
  exit 0
}

New-Item -ItemType Directory -Force -Path $binDir | Out-Null

$tmp = Join-Path $env:TEMP ('ffmpeg_dl_' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$zip = Join-Path $tmp 'ffmpeg.zip'

# 主源：BtbN/FFmpeg-Builds（latest 固定 tag）；备源：gyan.dev
$urls = @(
  'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip',
  'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
)

$downloaded = $false
foreach ($u in $urls) {
  try {
    Write-Host "[down] $u"
    Invoke-WebRequest -Uri $u -OutFile $zip -UseBasicParsing -TimeoutSec 900
    $downloaded = $true
    break
  } catch {
    Write-Host "[warn] 该下载源失败：$($_.Exception.Message)"
  }
}

if (-not $downloaded) {
  Write-Host "[error] ffmpeg 自动下载失败。"
  Write-Host "        请手动下载 Windows 静态构建，把 ffmpeg.exe 与 ffprobe.exe 放入："
  Write-Host "        $binDir"
  Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
  exit 1
}

try {
  Expand-Archive -Path $zip -DestinationPath $tmp -Force
} catch {
  Write-Host "[error] 解压失败：$($_.Exception.Message)"
  Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
  exit 1
}

$found = Get-ChildItem -Path $tmp -Recurse -Filter 'ffmpeg.exe' -File | Select-Object -First 1
if (-not $found) {
  Write-Host "[error] 解压后未找到 ffmpeg.exe"
  Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
  exit 1
}

Copy-Item $found.FullName $ffmpeg -Force
$probeSrc = Join-Path $found.DirectoryName 'ffprobe.exe'
if (Test-Path $probeSrc) { Copy-Item $probeSrc $ffprobe -Force }

Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "[ok] ffmpeg 已就位：$binDir"
exit 0
