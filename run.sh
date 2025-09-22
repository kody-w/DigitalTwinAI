#!/bin/bash
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi
echo "Starting Copilot Agent 365..."
echo "Local URL: http://localhost:7071/api/businessinsightbot_function"
echo "Press Ctrl+C to stop"
func start
