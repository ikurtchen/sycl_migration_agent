#!/bin/bash

# CUDA Vector Addition Build Script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== CUDA Vector Addition Build Script ===${NC}"

# Check if CUDA is installed
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: NVCC (CUDA compiler) not found in PATH${NC}"
    echo -e "${YELLOW}Please install CUDA Toolkit and ensure it's in your PATH${NC}"
    exit 1
fi

# Display CUDA version
echo -e "${GREEN}CUDA Version:${NC}"
nvcc --version | grep "release"
echo ""

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Cleaning existing build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

echo -e "${GREEN}Creating build directory...${NC}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo -e "${GREEN}Configuring with CMake...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release

# Check if configuration was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
fi

# Build the project
echo -e "${GREEN}Building the project...${NC}"
make -j$(nproc)

# Check if build was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}=== Build Successful! ===${NC}"
echo -e "${GREEN}Executable location: $(pwd)/bin/vectorAdd_cuda${NC}"

# Optional: Run the executable
read -p "Do you want to run the vectorAdd_cuda executable? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Running vectorAdd_cuda...${NC}"
    ./bin/vectorAdd_cuda
fi

cd ".."
echo -e "${GREEN}Done!${NC}"