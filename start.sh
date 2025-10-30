#!/bin/bash
# Start the FastAPI server
uvicorn unified_api:app --host 0.0.0.0 --port ${PORT:-8001}
