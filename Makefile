CC = nvcc

DEPS = utils.cu v0.cu v1.cu v2.cu main.cu
TEST_DEPS = utils.cu test.cu

all: main test

main: $(DEPS)
	$(CC) $(DEPS) -o main

test: $(TEST_DEPS)
	$(CC) $(TEST_DEPS) -o test

clean: 
	rm -rf main test
