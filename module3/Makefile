
NVCC = nvcc
CXXFLAGS = -O3 -arch=sm_60 -std=c++17
TARGET = assignment.exe

all: $(TARGET)

$(TARGET): assignment.cu
	$(NVCC) $(CXXFLAGS) -o $(TARGET) assignment.cu

clean:
	rm -f $(TARGET) results.csv

.PHONY: all clean