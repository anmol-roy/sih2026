#!/usr/bin/env bash
# Add src/ to PYTHONPATH so uvicorn can find api.main
export PYTHONPATH="/opt/render/project/src:${PYTHONPATH}"
exec uvicorn api.main:app --host 0.0.0.0 --port $PORT
