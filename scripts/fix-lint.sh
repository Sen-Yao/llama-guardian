#!/bin/bash
# 快速修复脚本 - 自动修复 lint 错误

set -e

echo "🔧 Auto-fixing lint errors..."

# 1. Ruff fix
python3 -m ruff check src/ tests/ --fix

# 2. Ruff format
python3 -m ruff format src/ tests/

# 3. Verify
python3 -m ruff check src/ tests/

echo "✅ All lint errors fixed!"
echo ""
echo "Now you can commit:"
echo "  git add -A && git commit -m 'fix: lint errors'"

exit 0