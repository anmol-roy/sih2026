#!/usr/bin/env bash
# Add src/ to PYTHONPATH so uvicorn can find api.main
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
exec uvicorn api.main:app --host 0.0.0.0 --port $PORT
