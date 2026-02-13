#!/bin/bash

# Setup script for The Auto-Grader project
# This script sets up the environment and generates the initial dataset

echo "================================================"
echo "The Auto-Grader - Setup Script"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "Found: $python_version"
echo ""

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (recommended) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "Virtual environment activated!"
    echo ""
fi

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed successfully!"
echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p data
mkdir -p models
mkdir -p results
mkdir -p logs
echo "✓ Directories created!"
echo ""

# Generate training dataset
echo "Generating training dataset..."
cd data
python generate_dataset.py
cd ..
echo "✓ Dataset generated successfully!"
echo ""

# Display summary
echo "================================================"
echo "Setup Complete! ✓"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Review the generated dataset: data/train_dataset.json"
echo "  2. Train the model: python src/train.py"
echo "  3. Or use Google Colab: notebooks/auto_grader_colab.ipynb"
echo ""
echo "For detailed instructions, see QUICKSTART.md"
echo ""
