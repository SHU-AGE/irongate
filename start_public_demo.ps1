# IronGate 公网演示一键启动
#   1) 本地若没跑服务就先起 python -m irongate serve
#   2) 用 SSH 反向隧道把它映射到 https://<随机>.lhr.life
#   3) 断开自动重连（注意：每次重连公网地址会变，脚本会打印新地址）
#
# 用法：右键“使用 PowerShell 运行”，或在终端里：
#   pwsh -File start_public_demo.ps1
# 可选参数：
#   -Port 8710           本地端口
#   -ProjectDir "D:\..." 项目根目录
#   -Python "D:\python\python.exe"

param(
    [int]$Port = 8710,
    [string]$ProjectDir = 'D:\新建文件夹 (14)\irongate',
    [string]$Python = 'D:\python\python.exe'
)

$ErrorActionPreference = 'Stop'
$localUrl = "http://127.0.0.1:$Port"

function Test-Local {
    try {
        $r = Invoke-WebRequest -Uri "$localUrl/api/health" -UseBasicParsing -TimeoutSec 3
        return $r.StatusCode -eq 200
    } catch { return $false }
}

Write-Host "== IronGate 公网演示 ==" -ForegroundColor Cyan

if (-not (Test-Local)) {
    if (-not (Test-Path $ProjectDir)) { throw "找不到项目目录：$ProjectDir" }
    if (-not (Test-Path $Python)) { throw "找不到 Python：$Python" }
    Write-Host "本地服务未运行，正在启动（端口 $Port）..."
    Start-Process -FilePath $Python `
        -ArgumentList '-m', 'irongate', 'serve', '--host', '127.0.0.1', '--port', $Port `
        -WorkingDirectory $ProjectDir -WindowStyle Hidden
    $ok = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-Local) { $ok = $true; break }
    }
    if (-not $ok) { throw "本地服务 20 秒内没起来，先在项目目录手动运行 python -m irongate serve --port $Port 看看报什么错" }
}
Write-Host "本地服务 OK：$localUrl" -ForegroundColor Green

$urlFile = Join-Path $PSScriptRoot 'public_url.txt'
$knownHosts = Join-Path $env:TEMP 'irongate_known_hosts'
$current = ''

# ssh 会把 "Creating known_hosts..." 之类写到 stderr，PowerShell 会当成错误记录；
# 这里切成 Continue，免得一次无害的警告把整个脚本掀翻。
$ErrorActionPreference = 'Continue'
while ($true) {
    Write-Host "建立隧道中..." -ForegroundColor DarkGray
    ssh -o StrictHostKeyChecking=accept-new `
        -o "UserKnownHostsFile=$knownHosts" `
        -o BatchMode=yes `
        -o ServerAliveInterval=10 `
        -o ServerAliveCountMax=6 `
        -o TCPKeepAlive=yes `
        -R "80:127.0.0.1:$Port" nokey@localhost.run 2>&1 | ForEach-Object {
        $line = "$_"
        if ($line -match 'https://([a-z0-9]+)\.lhr\.life') {
            if ($matches[0] -ne $current) {
                $current = $matches[0]
                Set-Content -Path $urlFile -Value $current -Encoding ASCII
                Write-Host ''
                Write-Host "  公网地址：$current" -ForegroundColor Yellow
                Write-Host "  （可直接发给别人；本窗口关掉就失效）" -ForegroundColor DarkGray
                Write-Host ''
            }
        }
    }
    Write-Host "隧道断开，3 秒后重连（地址会变，注意看新地址）..." -ForegroundColor DarkYellow
    Start-Sleep -Seconds 3
}
