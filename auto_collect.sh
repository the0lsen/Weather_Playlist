#!/bin/bash
# auto_collect.sh
# Use ./auto_collect.sh & to run it as long as terminal is open
# Runs collect.py every 30 minutes in the background.
# Logs output to auto_collect.log in the same directory.

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/auto_collect.log"
INTERVAL=1800  # 30 minutes in seconds

echo " Auto-collector started. Logging to $LOG"
echo "   Press Ctrl+C to stop."
echo ""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] Running collect.py..." | tee -a "$LOG"
    cd "$DIR" && python collect.py >> "$LOG" 2>&1
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] Done. Sleeping 30 minutes..." | tee -a "$LOG"
    echo "" >> "$LOG"
    sleep $INTERVAL
done

