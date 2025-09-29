#!/bin/bash
# Development setup script for Audio-Visual Speaker Detection System

echo "🛠️  Setting up Audio-Visual Speaker Detection Development Environment"
echo "=================================================================="

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "   $python_version"

if [[ $python_version == *"3.8"* ]] || [[ $python_version == *"3.9"* ]] || [[ $python_version == *"3.10"* ]] || [[ $python_version == *"3.11"* ]] || [[ $python_version == *"3.12"* ]]; then
    echo "✅ Python version is compatible"
else
    echo "⚠️  Warning: Python 3.8+ recommended"
fi

echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "Vision" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv Vision
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source Vision/bin/activate

echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
echo "📚 Installing development dependencies..."
pip install -r requirements.txt

echo ""

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p logs
mkdir -p config
mkdir -p tests
mkdir -p scripts

echo "✅ Project directories created"

echo ""

# Run initial tests
echo "🧪 Running initial system tests..."
python main.py --test

if [ $? -eq 0 ]; then
    echo "✅ System tests passed!"
else
    echo "⚠️  Some system tests failed - check hardware connections"
fi

echo ""

# List available devices
echo "🔍 Listing available devices..."
python main.py --list-devices

echo ""

echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Update device indexes in config/default.yaml if needed"
echo "  2. Run 'python main.py' to start the system"
echo "  3. Run './scripts/run_tests.sh' to run the full test suite"
echo "  4. Run 'python main.py --verbose' for debug mode"
echo ""
echo "Development commands:"
echo "  - Run tests: ./scripts/run_tests.sh"
echo "  - Format code: black src/ tests/ main.py"
echo "  - Lint code: flake8 src/ tests/ main.py"
echo "  - Run with coverage: pytest tests/ --cov=src"
echo ""
echo "Happy coding! 🚀"
