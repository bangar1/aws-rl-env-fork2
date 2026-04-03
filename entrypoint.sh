#!/bin/sh

# Trap SIGTERM/SIGINT so both processes get cleaned up on container stop
cleanup() {
    echo "Shutting down..."
    kill "$MINISTACK_PID" 2>/dev/null
    kill "$RL_SERVER_PID" 2>/dev/null
    wait "$MINISTACK_PID" 2>/dev/null
    wait "$RL_SERVER_PID" 2>/dev/null
    exit 0
}
trap cleanup TERM INT

# Start MiniStack (AWS emulator) in background on port 4566
echo "Starting MiniStack (AWS emulator) on port 4566..."
cd /app/env
python -m uvicorn ministack.app:app --host 0.0.0.0 --port 4566 --log-level info &
MINISTACK_PID=$!

# Wait for MiniStack to be healthy before starting the RL server
echo "Waiting for MiniStack to become healthy..."
RETRY=0
while [ $RETRY -lt 30 ]; do
    if python -c "import urllib.request; urllib.request.urlopen('http://localhost:4566/_ministack/health')" 2>/dev/null; then
        echo "MiniStack is ready."
        break
    fi
    RETRY=$((RETRY + 1))
    sleep 1
done

if [ $RETRY -eq 30 ]; then
    echo "WARNING: MiniStack did not become healthy in 30s, starting RL server anyway..."
fi

# Start RL Environment server in background on port 8000
echo "Starting RL Environment server on port 8000..."
uvicorn server.app:app --host 0.0.0.0 --port 8000 &
RL_SERVER_PID=$!

# Wait on both processes — if either exits, clean up the other
wait -n "$MINISTACK_PID" "$RL_SERVER_PID" 2>/dev/null
cleanup
