#!/bin/bash

# Installation script for Mujoco_View_Robot
# This script installs all required dependencies

set -e  # Exit on error

echo "=========================================="
echo "Installing Mujoco_View_Robot Dependencies"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed. Please install Python3 first."
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Check if pip is available
if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
    echo "Error: pip is not installed. Please install pip first."
    exit 1
fi

# Determine pip command
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
else
    PIP_CMD="python3 -m pip"
fi

echo "Using pip: $PIP_CMD"
echo ""

# Install dependencies
echo "Installing Python dependencies..."
echo "----------------------------------"

# Core dependencies
$PIP_CMD install --upgrade pip

echo ""
echo "Installing mujoco..."
$PIP_CMD install mujoco

echo ""
echo "Installing PyYAML..."
$PIP_CMD install pyyaml

echo ""
echo "Installing pyzmq..."
$PIP_CMD install pyzmq

echo ""
echo "Installing protobuf..."
$PIP_CMD install protobuf

echo ""
echo "Installing numpy..."
$PIP_CMD install numpy

echo ""
echo "=========================================="
echo "Installation completed successfully!"
echo "=========================================="
echo ""
echo "To run the live FK visualization:"
echo "  cd scripts"
echo "  python3 live_mujoco.py --ip 192.168.2.30"
echo ""
echo "Optional: Install additional packages"
echo "  - For better visualization: pip install mujoco-viewer"
echo ""
