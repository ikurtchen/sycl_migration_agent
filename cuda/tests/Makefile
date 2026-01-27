# Makefile for CUDA VectorAdd Tests

# Compiler and flags
NVCC = nvcc
CXX = g++

# CUDA and GTest paths (adjust if needed)
CUDA_PATH = /usr/local/cuda
GTEST_PATH = /usr/src/gtest

# Compiler flags
CUDA_FLAGS = -std=c++14 -arch=sm_60 -Xcompiler -fopenmp
CXX_FLAGS = -std=c++14 -O2

# Include paths
INCLUDES = -I$(CUDA_PATH)/include -I$(GTEST_PATH)/include -I.

# Library paths
LIBPATHS = -L$(CUDA_PATH)/lib64

# Libraries
LIBS = -lcudart -lcppunit -lgtest -lgtest_main -lpthread

# Sources
SOURCES = test_vectorAdd.cpp vectorAdd_kernel.cu
OBJECTS = $(SOURCES:.cpp=.o)
OBJECTS := $(OBJECTS:.cu=.o)

# Target executable
TARGET = vectorAdd_test

# Default target
all: $(TARGET)

# Build the test executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $@ $(LIBPATHS) $(LIBS)

# Compile C++ files
%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA files
%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@

# Build and run tests
test: $(TARGET)
	./run_tests.sh

# Clean build artifacts
clean:
	rm -f *.o $(TARGET)
	rm -rf build
	rm -rf cuda_outputs/*
	rm -rf test_logs/*

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install -y nvidia-cuda-toolkit libgtest-dev cmake
	cd /usr/src/gtest && sudo cmake . && sudo make && sudo make install
	sudo ln -s /usr/lib/libgtest.a /usr/lib/libgtest_main.a /usr/local/lib/

# Show help
help:
	@echo "Available targets:"
	@echo "  all          - Build the test executable"
	@echo "  test         - Build and run tests"
	@echo "  clean        - Remove build artifacts"
	@echo "  install-deps - Install required dependencies"
	@echo "  help         - Show this help message"

.PHONY: all test clean install-deps help