#!/bin/bash
# EC2 (Amazon Linux 2023) bootstrap: installs Docker, clones the repo, builds and runs the app on port 80.
# Bedrock credentials come from the instance's IAM role - no keys on the box.
set -euxo pipefail

REPO_URL="https://github.com/saurabhgoyal795/Capstone-Project.git"
APP_DIR=/opt/complaint-app

dnf install -y docker git
systemctl enable --now docker

git clone "$REPO_URL" "$APP_DIR"
cp "$APP_DIR/deploy/redeploy.sh" /usr/local/bin/redeploy-complaint-app
chmod +x /usr/local/bin/redeploy-complaint-app
/usr/local/bin/redeploy-complaint-app
