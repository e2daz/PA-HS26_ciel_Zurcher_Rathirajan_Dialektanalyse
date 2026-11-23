#!/bin/bash
# Auto-deploy for the dialect-analysis Streamlit viz.
# Installed as /usr/local/sbin/dialect-deploy (root-owned) and run every minute by
# /etc/cron.d/dialect-deploy. The cron job runs the INSTALLED copy, never this file in the repo.
# See deploy/DEPLOY.md, section "Automatic deploys".
set -euo pipefail
exec 9>/run/dialect-deploy.lock
flock -n 9 || exit 0            # skip if the previous run is still going

REPO=/opt/dialect-analysis
app() { sudo -u dialectanalysis -H "$@"; }

app git -C "$REPO" fetch -q origin main
OLD=$(app git -C "$REPO" rev-parse HEAD)
NEW=$(app git -C "$REPO" rev-parse origin/main)
[ "$OLD" = "$NEW" ] && exit 0

echo "$(date -Is) deploying ${OLD:0:7} -> ${NEW:0:7}"
app git -C "$REPO" merge --ff-only -q origin/main

changed() { ! app git -C "$REPO" diff --quiet "$OLD" "$NEW" -- "$1"; }

if changed requirements.txt; then
  echo "requirements.txt changed, installing dependencies"
  app sh -c "cd $REPO && uv pip install -r requirements.txt"
fi
if changed deploy/streamlit.service; then
  echo "streamlit.service changed, reloading unit"
  cp "$REPO/deploy/streamlit.service" /etc/systemd/system/
  systemctl daemon-reload
fi
if changed deploy/dialect-deploy.sh; then
  echo "NOTE: deploy/dialect-deploy.sh changed; reinstall it manually (see DEPLOY.md)"
fi

systemctl restart streamlit
sleep 5
if systemctl is-active --quiet streamlit; then
  echo "$(date -Is) ok, streamlit active"
else
  echo "$(date -Is) FAILED, streamlit not active after restart"
  journalctl -u streamlit -n 30 --no-pager
  exit 1
fi
