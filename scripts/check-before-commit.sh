#!/bin/bash
# 本地 CI 检查脚本 - 在提交前运行，避免 GitHub CI failure

set -e

echo "🔍 Running local CI checks..."

# 1. Ruff lint check
echo "1️⃣ Ruff lint check..."
python3 -m ruff check src/ tests/ --output-format=concise
if [ $? -ne 0 ]; then
    echo "❌ Ruff check failed! Run: python3 -m ruff check src/ tests/ --fix"
    exit 1
fi
echo "✅ Ruff check passed"

# 2. Ruff format check
echo "2️⃣ Ruff format check..."
python3 -m ruff format --check src/ tests/
if [ $? -ne 0 ]; then
    echo "❌ Format check failed! Run: python3 -m ruff format src/ tests/"
    exit 1
fi
echo "✅ Format check passed"

# 3. Import test (确保所有模块可导入)
echo "3️⃣ Import test..."
python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from config import load_config
    from vram_scheduler import VRamScheduler
    from whisper_manager import WhisperManager
    from task_queue import TaskQueue
    from stt_router import handle_stt_request
    print('All imports OK')
except Exception as e:
    print(f'Import failed: {e}')
    sys.exit(1)
"
if [ $? -ne 0 ]; then
    echo "❌ Import test failed!"
    exit 1
fi
echo "✅ Import test passed"

# 4. Syntax check (所有 Python 文件)
echo "4️⃣ Syntax check..."
python3 -m py_compile src/*.py tests/*.py 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Syntax check failed!"
    exit 1
fi
echo "✅ Syntax check passed"

echo ""
echo "🎉 All checks passed! Safe to commit."
echo ""
echo "Usage:"
echo "  ./scripts/check-before-commit.sh    # Run checks"
echo "  git add -A && git commit -m '...'   # Then commit"
echo ""

exit 0