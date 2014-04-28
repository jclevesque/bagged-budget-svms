CXX ?= g++
CFLAGS = -g -Wall -Wconversion -O3
SHVER = 2
OS = $(shell uname)
dir_guard=@mkdir -p $(@D)
OUT_DIR = bin

all: $(OUT_DIR)/budgetedsvm-train $(OUT_DIR)/budgetedsvm-predict 
		
$(OUT_DIR)/budgetedsvm-train: src/budgetedsvm-train.cpp mm_algs.o bsgd.o llsvm.o budgetedSVM.o
	$(dir_guard)
	$(CXX) $(CFLAGS) src/budgetedsvm-train.cpp mm_algs.o bsgd.o llsvm.o budgetedSVM.o -o $(OUT_DIR)/budgetedsvm-train -lm
$(OUT_DIR)/budgetedsvm-predict: src/budgetedsvm-predict.cpp mm_algs.o bsgd.o llsvm.o budgetedSVM.o
	$(dir_guard)
	$(CXX) $(CFLAGS) src/budgetedsvm-predict.cpp mm_algs.o bsgd.o llsvm.o budgetedSVM.o -o $(OUT_DIR)/budgetedsvm-predict -lm
	
mm_algs.o: src/mm_algs.cpp src/mm_algs.h
	$(CXX) $(CFLAGS) -c src/mm_algs.cpp
bsgd.o: src/bsgd.cpp src/bsgd.h
	$(CXX) $(CFLAGS) -c src/bsgd.cpp
llsvm.o: src/llsvm.cpp src/llsvm.h
	$(CXX) $(CFLAGS) -c src/llsvm.cpp
budgetedSVM.o: src/budgetedSVM.cpp src/budgetedSVM.h
	$(CXX) $(CFLAGS) -c src/budgetedSVM.cpp
clean:
	rm -f *~ budgetedSVM.o bsgd.o mm_algs.o llsvm.o
