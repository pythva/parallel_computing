#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../include/parallelHeader.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define block_size 1024
#define MIN_SCRATCH_SIZE (50UL * 1024UL * 1024UL) // 50 MB

/** The global huffman dictionary, used in compression to map values */
extern struct huffmanDictionary huffmanDictionary;

// We expose this function to the C/C++ linker so MPI code can call it.
extern "C" {
void runCudaLand(
    int myrank,
    unsigned char *inputFileData,
    uint64_t inputFileLength,
    unsigned char **hostCompressedData,
    uint64_t hostCompressedSize
);
} // extern "C"

/**
 * @brief Does the cuda side of things to compress each chunk
 *
 * @param myrank
 * @param inputFileData The uncompressed input chunk (note the code seems to
 * change with the data in here)
 * @param inputFileLength The length of the uncompressed input chunk
 * @param hostCompressedData [out] The newly-allocated byte-aligned compressed
 * output
 * @param hostCompressedSize The expected compressed size, in bits
 */
void runCudaLand(
    int myrank,
    unsigned char *inputFileData,
    uint64_t inputFileLength,
    unsigned char **hostCompressedData,
    uint64_t hostCompressedSize
) {
    // Optionally choose device if multiple GPUs are present
    // e.g., if you have 4 GPUs per node, do:
    cudaSetDevice(myrank % 4);

    // Local variables
    uint64_t i;
    uint64_t mem_free, mem_total;
    uint64_t mem_data, mem_req;
    uint64_t compressed_size_bytes =
        hostCompressedSize / 8 + (hostCompressedSize % 8 != 0);
    int numKernelRuns;
    unsigned int integerOverflowFlag = 0;

    // --------------------------------------------------------------------
    // Calculate memory requirements (as in your original code)
    // --------------------------------------------------------------------
    cudaMemGetInfo(&mem_free, &mem_total);
    printf(
        "Rank %d: GPU free mem: %lu (of %lu)\n",
        myrank,
        mem_free,
        mem_total
    );

    // offset array requirements: total bits
    uint64_t mem_offset = hostCompressedSize;
    if (mem_offset % 8 != 0) {
        mem_offset += (8 - (mem_offset % 8));
    }

    // other memory usage
    //   inputFileLength bytes for data
    //   (inputFileLength + 1) * sizeof(unsigned int) for offset array
    //   plus dictionary overhead
    mem_data = inputFileLength + (inputFileLength + 1) * sizeof(uint64_t) +
               sizeof(huffmanDictionary);

    // check if we have enough free memory
    if (mem_free <= mem_data + MIN_SCRATCH_SIZE) {
        printf(
            "Rank %d: Not enough GPU memory for compression.\n"
            "        mem_free=%lu, need at least %lu\n",
            myrank,
            mem_free,
            mem_data + MIN_SCRATCH_SIZE
        );
        // In your original code, you'd do "return -1;"
        // But here we can't do that exactly because function is void.
        // We can set *hostCompressedData=NULL and *hostCompressedSize=0, then
        // return
        *hostCompressedData = NULL;
        return;
    }

    // mem_req is how much scratch we can actually use
    mem_req = mem_free - mem_data - (10UL * 1024UL * 1024UL);

    // figure out how many kernel runs if the offset is bigger than mem_req
    numKernelRuns = (int)ceil((double)mem_offset / (double)mem_req);

    // detect integer overflow corner case
    // (your original code checks if mem_req + 255 <= UINT_MAX, etc.)
    // adapted version here:
    if ((mem_req + 255) > UINT64_MAX || (mem_offset + 255) > UINT64_MAX) {
        integerOverflowFlag = 1;
    } else {
        integerOverflowFlag = 0;
    }

    printf(
        "Rank %d: InputFileSize=%zu, OutputSize=%zu, NumberOfKernel=%d, "
        "OverflowFlag=%d\n",
        myrank,
        inputFileLength,
        mem_offset / 8,
        numKernelRuns,
        integerOverflowFlag
    );

    // --------------------------------------------------------------------
    // Allocate compressedDataOffset array on CPU
    // (like your original code does)
    // --------------------------------------------------------------------
    uint64_t *compressedDataOffset =
        (uint64_t *)malloc((inputFileLength + 1) * sizeof(uint64_t));
    if (!compressedDataOffset) {
        fprintf(
            stderr,
            "Rank %d: Failed to allocate compressedDataOffset.\n",
            myrank
        );
        *hostCompressedData = NULL;
        return;
    }

    lauchCUDAHuffmanCompress(
        inputFileData,        // pointer to your local chunk
        compressedDataOffset, // offset array
        inputFileLength,      // length
        numKernelRuns,
        integerOverflowFlag,
        mem_req
    );

    // --------------------------------------------------------------------
    // At this point, your original code wrote the compressed bits into
    // inputFileData at the front. The total # of compressed bytes is
    // mem_offset/8. We'll allocate a new buffer for the output and copy from
    // inputFileData.
    // --------------------------------------------------------------------

    // allocate host output buffer
    unsigned char *tmpOutput =
        (unsigned char *)malloc(compressed_size_bytes * sizeof(unsigned char));
    if (!tmpOutput) {
        fprintf(stderr, "Rank %d: Failed to allocate output buffer.\n", myrank);
        free(compressedDataOffset);
        *hostCompressedData = NULL;
        return;
    }

    // copy from inputFileData (the front part of it) to tmpOutput
    memcpy(tmpOutput, inputFileData, compressed_size_bytes);

    // set the returned pointers
    *hostCompressedData = tmpOutput;

    // cleanup
    free(compressedDataOffset);

    printf(
        "Rank %d: runCudaLand done. Compressed size=%zu bytes\n",
        myrank,
        compressed_size_bytes
    );
}
