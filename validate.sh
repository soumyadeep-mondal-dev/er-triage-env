#!/usr/bin/env bash

SPACE_URL=$1
REPO_DIR=${2:-.}

if [ -z "$SPACE_URL" ]; then
  echo "Usage: ./validate.sh <space_url> [repo_dir]"
  exit 1
fi

echo "========================================"
echo " OpenEnv Pre-Submission Validator"
echo "========================================"
echo "Space URL: $SPACE_URL"
echo ""

PASS_COUNT=0

# ── CHECK DEPENDENCIES ─────────────────────────────
command -v docker >/dev/null 2>&1 || { echo "❌ Docker not installed"; exit 1; }
command -v openenv >/dev/null 2>&1 || { echo "❌ openenv not installed"; exit 1; }

# ── STEP 1: HF SPACE ───────────────────────────────
echo "[1/3] Checking HF Space /reset..."

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" \
  -d '{}' \
  --max-time 15 \
  "$SPACE_URL/reset")

if [ "$HTTP_CODE" = "200" ]; then
  echo "✅ PASS: HF Space responding"
  ((PASS_COUNT++))
else
  echo "❌ FAIL: /reset returned HTTP $HTTP_CODE"
  exit 1
fi

# ── STEP 2: DOCKER BUILD ───────────────────────────
echo ""
echo "[2/3] Checking Docker build..."

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  echo "❌ FAIL: No Dockerfile found"
  exit 1
fi

echo "Building Docker (this may take time)..."
docker build "$DOCKER_CONTEXT"

if [ $? -eq 0 ]; then
  echo "✅ PASS: Docker build successful"
  ((PASS_COUNT++))
else
  echo "❌ FAIL: Docker build failed"
  exit 1
fi

# ── STEP 3: OPENENV VALIDATE ───────────────────────
echo ""
echo "[3/3] Running openenv validate..."

openenv validate

if [ $? -eq 0 ]; then
  echo "✅ PASS: openenv validate successful"
  ((PASS_COUNT++))
else
  echo "❌ FAIL: openenv validate failed"
  exit 1
fi

# ── FINAL RESULT ───────────────────────────────────
echo ""
echo "========================================"
echo "Passed: $PASS_COUNT / 3"
echo "🎯 ALL CHECKS PASSED"
echo "SUCCESSFULLY VALIDATED 🚀"
echo "========================================"