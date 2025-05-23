###############################################################################
# Makefile for building an MPI + CUDA program
# -------------------------------------------
# - We compile C files with mpixlc (MPI compiler).
# - We compile CUDA files with nvcc.
# - Finally, we link everything into a single executable: mpi-cuda-exe.
#
# Edit the variables below (MPICC, NVCC, CUDALIB, etc.) to match your paths.
###############################################################################

# 1) Choose compilers and flags:
MPICC      = mpixlc               # IBM MPI compiler, or use mpicc, etc.
NVCC       = nvcc
CFLAGS     = -O3 -m64
NVCCFLAGS  = -O3 -arch=sm_70 -m64      # Adjust your SM architecture (e.g. sm_70)
CUDALIB    = -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart

# 3) List your sources
#    - new_compress.c        (the MPI code)
#    - serialFunctions.c     (some CPU Huffman / utility code?)
#    - mpi-cuda.cu           (the "bridge" or combined MPI+CUDA code)
#    - parallelFunctions.cu, kernel.cu, GPUWrapper.cu (your existing CUDA code)
#
#    We'll produce corresponding .o files for each:
CSRCS_C    = new_compress.c ../../include/parallelFunctionsHost.c
CSRCS_D    = decompress.c ../../include/parallelFunctionsHost.c
# CUSRCS     = runCudaLand.cu ../../include/parallelFunctions.cu ../../include/kernel.cu ../../include/GPUWrapper.cu
CUSRCS     = cuda_new_compress.cu

# 4) Derive object-file names from those sources
COBJS_C    = $(CSRCS_C:.c=.o)
COBJS_D    = $(CSRCS_D:.c=.o)
CUOBJS     = $(CUSRCS:.cu=.o)

###############################################################################
# The default rule: build everything
###############################################################################
all: compress decompress

###############################################################################
# How to link the final executable
###############################################################################
compress: $(COBJS_C) $(CUOBJS)
	# Use mpixlc to link everything.  Link with CUDA libs and stdc++ as needed.
	$(MPICC) $(CFLAGS) $^ -o $@ $(CUDALIB) -lstdc++
decompress: $(COBJS_D)
	$(MPICC) $(CFLAGS) $^ -o $@ $(CUDALIB) -lstdc++

###############################################################################
# Compile rules
#
# 1) Compile .c -> .o with mpixlc
# 2) Compile .cu -> .o with nvcc
###############################################################################

# Rule for C sources
%.o: %.c
	$(MPICC) $(CFLAGS) -c $< -o $@

# Rule for CUDA sources
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

###############################################################################
# Clean up
###############################################################################
clean:
	rm -f $(COBJS) $(CUOBJS) $(TARGET)

