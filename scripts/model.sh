#!/usr/bin/env bash
# CLI to call the Modal endpoints directly. Requires .env to be sourced or
# exported (MODAL_*_URL vars).
set -euo pipefail

cmd="${1:-}"
shift || true

case "$cmd" in
  detect)
    curl -s -X POST "$MODAL_DETECT_URL" -H "Content-Type: application/json" \
      -d "{\"text\": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"), \"model\": \"desklib/ai-text-detector-v1.01\"}"
    ;;
  paraphrase)
    curl -s -X POST "$MODAL_PARAPHRASE_URL" -H "Content-Type: application/json" \
      -d "{\"text\": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"), \"strength\": 3}"
    ;;
  backtranslate)
    curl -s -X POST "$MODAL_BACKTRANSLATE_URL" -H "Content-Type: application/json" \
      -d "{\"text\": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1")}"
    ;;
  embed)
    curl -s -X POST "$MODAL_EMBED_URL" -H "Content-Type: application/json" \
      -d "{\"text\": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1")}"
    ;;
  health)
    curl -s "$MODAL_HEALTH_URL"
    ;;
  *)
    echo "Usage: model.sh {detect|paraphrase|backtranslate|embed|health} [text]"
    exit 1
    ;;
esac
echo
