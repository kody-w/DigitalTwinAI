#!/bin/bash
# Quick start script for the Azure Function

# Activate virtual environment
source .venv/bin/activate

# Start the function
echo "Starting Azure Function on http://localhost:7071"
echo "Press Ctrl+C to stop"
func start
