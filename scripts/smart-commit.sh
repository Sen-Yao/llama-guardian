#!/bin/bash
# 提交前自动检查 + 修复 + 提交

set -e

COMMIT_MSG="${1:-update}"

echo "🔍 Pre-commit check..."

# Run checks
./scripts/check-before-commit.sh

if [ $? -eq 0 ]; then
    echo "✅ Checks passed, committing..."
    git add -A
    git commit -m "$COMMIT_MSG"
    echo "✅ Committed: $COMMIT_MSG"
    echo ""
    echo "Push with: git push origin main"
else
    echo "❌ Checks failed, fixing..."
    ./scripts/fix-lint.sh
    git add -A
    git commit -m "$COMMIT_MSG"
    echo "✅ Fixed and committed"
fi

exit 0