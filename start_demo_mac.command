#!/bin/bash
# IronGate 演示一键启动（macOS）
#   在访达里双击本文件即可；也可以在终端里运行：
#     ./start_demo_mac.command                 # 本机演示 http://127.0.0.1:8100
#     ./start_demo_mac.command --port 8710     # 换端口
#     ./start_demo_mac.command --public        # 额外开公网地址（同 start_public_demo.ps1）
#
# 做的事：
#   1) 找一个 >= 3.10 的 Python（优先项目里的 .venv，其次系统 python3）；
#      都没有时，若装了 uv 就自动在项目里建 .venv，否则提示安装方法
#   2) 启动 python -m irongate serve 并自动打开浏览器
#   3) 加 --public 时，用 SSH 反向隧道映射到 https://<随机>.lhr.life，断开自动重连
#
# Windows 请继续用 start_public_demo.ps1。

set -u
cd "$(dirname "$0")" || exit 1

PORT=8100
PUBLIC=0
while [ $# -gt 0 ]; do
    case "$1" in
        --port) [ $# -ge 2 ] || { echo "--port 后面要跟端口号"; exit 2; }; PORT="$2"; shift 2 ;;
        --public) PUBLIC=1; shift ;;
        *) echo "未知参数：$1"; exit 2 ;;
    esac
done

pause_and_exit() {
    echo
    read -r -p "按回车键关闭窗口..." _
    exit "$1"
}

python_ok() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null
}

PY=""
for candidate in .venv/bin/python python3; do
    if command -v "$candidate" >/dev/null 2>&1 && python_ok "$candidate"; then
        PY="$candidate"
        break
    fi
done

if [ -z "$PY" ]; then
    if command -v uv >/dev/null 2>&1; then
        echo "没找到 3.10 以上的 Python，正在用 uv 在项目里安装（只需一次）..."
        uv venv --python 3.12 .venv || pause_and_exit 1
        PY=.venv/bin/python
    else
        echo "需要 Python 3.10 或更高版本，这台电脑上没找到。"
        echo "安装方法（任选其一）："
        echo "  · 到 https://www.python.org/downloads/ 下载安装"
        echo "  · 或在终端运行：brew install python@3.12"
        pause_and_exit 1
    fi
fi

LOCAL_URL="http://127.0.0.1:$PORT"
echo "== IronGate 演示 =="
echo "使用 Python：$("$PY" --version 2>&1)"

if curl -fsS "$LOCAL_URL/api/health" >/dev/null 2>&1; then
    echo "本地服务已经在运行：$LOCAL_URL"
    SERVER_PID=""
    open "$LOCAL_URL"
else
    echo "正在启动本地服务：$LOCAL_URL"
    "$PY" -m irongate serve --host 127.0.0.1 --port "$PORT" &
    SERVER_PID=$!
    trap '[ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null' EXIT
    for _ in $(seq 1 40); do
        curl -fsS "$LOCAL_URL/api/health" >/dev/null 2>&1 && break
        sleep 0.5
    done
    if ! curl -fsS "$LOCAL_URL/api/health" >/dev/null 2>&1; then
        echo "本地服务 20 秒内没起来，请看上面的报错信息。"
        pause_and_exit 1
    fi
    open "$LOCAL_URL"
fi

if [ "$PUBLIC" -eq 0 ]; then
    echo
    echo "演示页已在浏览器打开。关掉这个窗口（或按 Ctrl+C）即停止服务。"
    [ -n "$SERVER_PID" ] && wait "$SERVER_PID"
    exit 0
fi

URL_FILE="public_url.txt"
KNOWN_HOSTS="${TMPDIR:-/tmp}/irongate_known_hosts"
CURRENT=""
while true; do
    echo "建立公网隧道中..."
    while IFS= read -r line; do
        url=$(printf '%s\n' "$line" | grep -oE 'https://[a-z0-9]+\.lhr\.life' | head -n 1)
        if [ -n "$url" ] && [ "$url" != "$CURRENT" ]; then
            CURRENT="$url"
            printf '%s\n' "$CURRENT" > "$URL_FILE"
            echo
            echo "  公网地址：$CURRENT"
            echo "  （可直接发给别人；本窗口关掉就失效）"
            echo
        fi
    done < <(ssh -o StrictHostKeyChecking=accept-new \
                 -o "UserKnownHostsFile=$KNOWN_HOSTS" \
                 -o BatchMode=yes \
                 -o ServerAliveInterval=10 \
                 -o ServerAliveCountMax=6 \
                 -o TCPKeepAlive=yes \
                 -R "80:127.0.0.1:$PORT" nokey@localhost.run 2>&1)
    echo "隧道断开，3 秒后重连（地址会变，注意看新地址）..."
    sleep 3
done
