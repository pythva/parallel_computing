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

struct huffmanTree *head_huffmanTreeNode;
struct huffmanTree huffmanTreeNode[512];
struct huffmanDictionary huffmanDictionary;
unsigned char bitSequenceConstMemory[256][255];
unsigned int constMemoryFlag = 0;

// We expose this function to the C/C++ linker so MPI code can call it.
extern "C" {
/**
 * @brief Bridge function called by MPI code to compress a chunk of data on GPU.
 *
 * @param myrank    The MPI rank (used for cudaSetDevice(...) or debug).
 * @param inputData Pointer to the local (CPU) buffer of size 'blockLength'.
 * @param blockLength The length of the local chunk in bytes.
 * @param frequency  Pointer to frequency array [256].
 * @param d_compressedData [OUT] We allocate and return a new CPU buffer that
 * holds compressed data.
 * @param compBlockLength [OUT] The final compressed size in bytes.
 */
void runCudaLand(
    int myrank,
    unsigned char *inputData,
    uint64_t blockLength,
    uint64_t *frequency,
    unsigned char **d_compressedData,
    uint64_t *compBlockLength
);
} // extern "C"

/**
 * This function replicates the main logic from your original CUDACompress.cu,
 * except:
 *   1) It doesn’t open/read input files or write the final output file.
 *   2) It receives data & freq from the caller (MPI code) and returns the
 *      compressed data + length via pointers.
 *   3) It calls lauchCUDAHuffmanCompress(...) to do the parallel bit-packing.
 */
void runCudaLand(
    int myrank,
    unsigned char *inputFileData,       // local chunk
    uint64_t inputFileLength,           // chunk length
    uint64_t *frequency,                // global frequency[256]
    unsigned char **hostCompressedData, // out: newly allocated CPU buffer
    uint64_t *hostCompressedSize
) // out: final size
{
    // Optionally choose device if multiple GPUs are present
    // e.g., if you have 4 GPUs per node, do:
    cudaSetDevice(myrank % 4);

    // Local variables
    uint64_t i;
    unsigned int distinctCharacterCount = 0, combinedHuffmanNodes = 0;
    unsigned char bitSequence[255];
    unsigned char bitSequenceLength = 0;
    uint64_t mem_offset = 0;
    uint64_t mem_free, mem_total;
    uint64_t mem_data, mem_req;
    int numKernelRuns;
    unsigned int integerOverflowFlag = 0;

    // --------------------------------------------------------------------
    // Build the Huffman Tree on CPU (similar to your original code)
    // --------------------------------------------------------------------
    head_huffmanTreeNode = NULL;
    memset(huffmanTreeNode, 0, sizeof(huffmanTreeNode));

    // 1) initialize nodes of the Huffman tree
    for (i = 0; i < 256; i++) {
        if (frequency[i] > 0) {
            huffmanTreeNode[distinctCharacterCount].count = frequency[i];
            huffmanTreeNode[distinctCharacterCount].letter = i;
            huffmanTreeNode[distinctCharacterCount].left = NULL;
            huffmanTreeNode[distinctCharacterCount].right = NULL;
            distinctCharacterCount++;
        }
    }

    // 2) build the Huffman tree
    for (i = 0; i < distinctCharacterCount - 1; i++) {
        combinedHuffmanNodes = 2 * i;
        sortHuffmanTree(i, distinctCharacterCount, combinedHuffmanNodes);
        buildHuffmanTree(i, distinctCharacterCount, combinedHuffmanNodes);
    }

    // 3) if there's only one distinct character
    if (distinctCharacterCount == 1) {
        head_huffmanTreeNode = &huffmanTreeNode[0];
    }

    // 4) build Huffman dictionary
    // memset(huffmanDictionary, 0, sizeof(huffmanDictionary));
    // buildHuffmanDictionary(head_huffmanTreeNode, bitSequence,
    // bitSequenceLength);
    buildHuffmanDictionary(
        head_huffmanTreeNode,
        bitSequence,
        bitSequenceLength
    );
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
    mem_offset = 0;

    /*
    for (i = 0; i < 256; i++){
        // total bits = sum of freq[i] * bitSequenceLength[i]
        //mem_offset += (frequency[i]) * huffmanDictionary.bitSequenceLength;

    mem_offset += frequency[i] * huffmanDictionary.bitSequenceLength[i];
    }
    // round up to multiple of 8
    if (mem_offset % 8 != 0){
        mem_offset += (8 - (mem_offset % 8));
    }*/

    for (i = 0; i < inputFileLength; i++) {
        unsigned char symbol = inputFileData[i];
        // Use the single global huffmanDictionary to get the code length
        mem_offset += huffmanDictionary.bitSequenceLength[symbol];
    }
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
        *hostCompressedSize = 0;
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
        *hostCompressedSize = 0;
        return;
    }

    // --------------------------------------------------------------------
    // Launch kernel to do the actual compression (like your original code)
    //   lauchCUDAHuffmanCompress(...) is presumably in parallelFunctions.cu or
    //   GPUWrapper.cu
    // --------------------------------------------------------------------
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
    uint64_t totalCompressedBytes = (uint64_t)(mem_offset / 8);

    // allocate host output buffer
    unsigned char *tmpOutput =
        (unsigned char *)malloc(totalCompressedBytes * sizeof(unsigned char));
    if (!tmpOutput) {
        fprintf(stderr, "Rank %d: Failed to allocate output buffer.\n", myrank);
        free(compressedDataOffset);
        *hostCompressedData = NULL;
        *hostCompressedSize = 0;
        return;
    }

    // copy from inputFileData (the front part of it) to tmpOutput
    memcpy(tmpOutput, inputFileData, totalCompressedBytes);

    // set the returned pointers
    *hostCompressedData = tmpOutput;
    *hostCompressedSize = totalCompressedBytes;

    // cleanup
    free(compressedDataOffset);

    printf(
        "Rank %d: runCudaLand done. Compressed size=%u bytes\n",
        myrank,
        totalCompressedBytes
    );
}
