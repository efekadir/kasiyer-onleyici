#!/bin/bash
cd "$(dirname "$0")"
PORT=8765
(sleep 1; open "http://localhost:$PORT") &
echo "Odak Bekçisi çalışıyor → http://localhost:$PORT  (kapatmak için Ctrl+C)"
python3 -m http.server $PORT
