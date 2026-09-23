#!/bin/bash
# Pull the latest code, rebuild the image and restart the container (run on the EC2 host).
set -euxo pipefail

APP_DIR=/opt/complaint-app
cd "$APP_DIR"
git pull --ff-only
docker build -t complaint-app:latest .
docker rm -f complaint-app 2>/dev/null || true
docker run -d --name complaint-app --restart unless-stopped \
  -p 80:8501 \
  -e LLM_PROVIDER=bedrock \
  -e AWS_REGION=ap-south-1 \
  -e MAX_WORKERS=4 \
  complaint-app:latest
docker image prune -f
