#!/usr/bin/env bash
set -o errexit

echo "==> Installing Chrome..."
apt-get update -qq
apt-get install -y wget gnupg ca-certificates apt-transport-https

wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub \
  | gpg --dearmor \
  | tee /etc/apt/trusted.gpg.d/google-chrome.gpg > /dev/null

echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" \
  > /etc/apt/sources.list.d/google-chrome.list

apt-get update -qq
apt-get install -y google-chrome-stable

echo "==> Chrome version:"
google-chrome --version

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

echo "==> Build complete."
