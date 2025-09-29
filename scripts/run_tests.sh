#!/bin/bash
# Test runner script for Audio-Visual Speaker Detection System

echo "🧪 Running Audio-Visual Speaker Detection Test Suite"
echo "=================================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Consider running: source Vision/bin/activate"
fi

echo ""

# Run unit tests
echo "📋 Running unit tests..."
python -m pytest tests/ -v --tb=short

# Check test results
if [ $? -eq 0 ]; then
    echo "✅ All unit tests passed!"
else
    echo "❌ Some unit tests failed"
    exit 1
fi

echo ""

# Run system component tests
echo "🔧 Testing system components..."
python main.py --test

if [ $? -eq 0 ]; then
    echo "✅ System component tests passed!"
else
    echo "❌ System component tests failed"
    exit 1
fi

echo ""

# Run code quality checks (if tools are available)
echo "🔍 Running code quality checks..."

# Check if black is available
if command -v black &> /dev/null; then
    echo "📝 Checking code formatting with black..."
    black --check src/ tests/ main.py
    if [ $? -eq 0 ]; then
        echo "✅ Code formatting is correct"
    else
        echo "⚠️  Code formatting issues found (run 'black src/ tests/ main.py' to fix)"
    fi
else
    echo "⚠️  black not available, skipping formatting check"
fi

# Check if flake8 is available
if command -v flake8 &> /dev/null; then
    echo "🔍 Running linting with flake8..."
    flake8 src/ tests/ main.py
    if [ $? -eq 0 ]; then
        echo "✅ No linting issues found"
    else
        echo "⚠️  Linting issues found"
    fi
else
    echo "⚠️  flake8 not available, skipping linting check"
fi

echo ""
echo "🎉 Test suite completed!"
echo ""
echo "Next steps:"
echo "  - Run 'python main.py' to start the system"
echo "  - Run 'python main.py --list-devices' to check hardware"
echo "  - Run 'python main.py --verbose' for debug mode"
