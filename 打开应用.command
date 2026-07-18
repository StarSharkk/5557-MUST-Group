#!/bin/bash
cd "$(dirname "$0")"

STREAMLIT_BIN="/opt/anaconda3/bin/streamlit"

if [ ! -x "$STREAMLIT_BIN" ]; then
  STREAMLIT_BIN="streamlit"
fi

echo "正在启动 AI Intraday Quant Trading Simulator ..."
echo "首次加载可能需要几秒钟，请稍候。"
echo "关闭这个终端窗口即可停止应用。"
echo ""

"$STREAMLIT_BIN" run app.py
