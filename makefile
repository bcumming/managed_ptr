NVCC_FLAGS=-std=c++11
NVCC_FLAGS+=-arch=sm_60
NVCC_FLAGS+=-g -O2

HEADERS=managed_ptr.hpp

all : test

gtest.o : gtest.cpp gtest/gtest.hpp
	g++ -std=c++11 -O2 -g -c gtest.cpp

test : gtest.o test.cu ${HEADERS}
	nvcc ${NVCC_FLAGS} test.cu gtest.o -o test

clean :
	rm -f test
	rm -f *.o
