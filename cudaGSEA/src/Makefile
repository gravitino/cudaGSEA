# compilers and flags
CXX=g++
CXXFLAGS=-O3 -march=native -fPIC
NVCC=nvcc
NVCCFLAGS=-arch=sm_35 -O3 -std=c++11 -gencode arch=compute_30,code=compute_30 \
                                     -gencode arch=compute_35,code=compute_35 \
                                     -gencode arch=compute_52,code=compute_52

# search nvcc in the filesystem and set if CUDA_HOME not set
CUDA_HOME ?= $(shell echo $$(dirname $$(dirname $$(find -P / -name "nvcc" -print -quit 2>/dev/null))))
CUDA_LIB=targets/x86_64-linux/lib/

# search R.h and Rcpp.h in the filesystem and set R_INC/RCPP_INC if not set
R_INC ?= $(shell echo $$(dirname $$(find -P / -name "R.h" -print -quit 2>/dev/null)))
RCPP_INC ?= $(shell echo $$(dirname $$(find -P / -name "Rcpp.h" -print -quit 2>/dev/null)))

# object to build
OBJECTS=cudaGSEA_wrapper_cpp.o cudaGSEA_wrapper_cu.o

all: print_path_info cudaGSEA.so

print_path_info:

	@echo -e 'INFO: === PATH INFO ============================================='
	@echo -e 'INFO: The cudaGSEA package is build using the following paths:   '
	@echo -e 'INFO:                                                            '
	@echo -e 'INFO: path to CUDA:   \t CUDA_HOME=$(CUDA_HOME)'
	@echo -e 'INFO: path to R.h:    \t R_INC=$(R_INC)'
	@echo -e 'INFO: path to Rcpp.h: \t RCPP_INC=$(RCPP_INC)'
	@echo -e 'INFO:                                                            '
	@echo -e 'INFO: If these are not your desired paths, please alter the      '
	@echo -e 'INFO: global environment variables CUDA_HOME, R_INC AND RCPP_INC.'
	@echo -e 'INFO: ==========================================================='

cudaGSEA.so: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -shared $(OBJECTS) -L $(CUDA_HOME)/$(CUDA_LIB) -lcudart -o cudaGSEA.so

cudaGSEA_wrapper_cu.o: cudaGSEA_wrapper.cu cudaGSEA_wrapper.cuh
	$(CUDA_HOME)/bin/$(NVCC) $(NVCCFLAGS) cudaGSEA_wrapper.cu -c -Xcompiler="$(CXXFLAGS)" -o cudaGSEA_wrapper_cu.o

cudaGSEA_wrapper_cpp.o: cudaGSEA_wrapper.cpp cudaGSEA_wrapper.cuh
	$(CXX) $(CXXFLAGS) cudaGSEA_wrapper.cpp -c -lR -I $(R_INC) -I $(RCPP_INC) -std=c++11 -o cudaGSEA_wrapper_cpp.o

cudaGSEA: cudaGSEA.cu
	$(CUDA_HOME)/bin/$(NVCC) $(NVCCFLAGS) cudaGSEA.cu -o cudaGSEA

clean:
	rm -rf $(OBJECTS) cudaGSEA.so cudaGSEA *~
