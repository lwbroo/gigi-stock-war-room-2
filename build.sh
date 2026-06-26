#!/bin/bash
set -e
pip install -r requirements.txt
# Install Playwright Chromium browser; gracefully skip if not supported
python -m playwright install chromium 2>/dev/null && echo "✅ Playwright Chromium installed" || echo "⚠️  Playwright Chromium skipped — yfinance fallback will be used"
