#include <cuda.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Include your headers that have sortHuffmanTree, buildHuffmanTree, etc.
// e.g., #include "../include/parallelHeader.h"
#include "../include/parallelHeader.h"

// We’ll define the bridging function with C linkage:
extern "C" {
// The bridging function: we pass in the local block (inputData), size,
// the frequency array, etc., and return a newly allocated compressed buffer
// plus length.
void runCudaLand(
    int myrank,
    unsigned char *inputData,
    unsigned int blockLength,
    unsigned int *frequency,
    unsigned char *
        *d_compressedData, // [OUT] pointer to newly alloc’d CPU buffer
    unsigned int *compBlockLength
); // [OUT] final number of compressed bytes
}

// We can reuse the same global arrays you had in "CUDACompress.cu"
struct huffmanTree *head_huffmanTreeNode;
struct huffmanTree huffmanTreeNode[512];

// Suppose we keep a global dictionary array here too:
struct huffmanDictionary huffmanDictionary[256];

unsigned char bitSequenceConstMemory[256][255];
unsigned int constMemoryFlag = 0;

// This function is basically your old “main” from CUDACompress.cu,
// but adapted to be callable by MPI code.  We remove file I/O and
// pass data in/out through function arguments.
void runCudaLand(
    int myrank,
    unsigned char *inputFileData,
    unsigned int inputSize,
    unsigned int *frequency,
    unsigned char **hostCompressedData, // [OUT]
    unsigned int *hostCompressedSize
) // [OUT]
{
    // Optional: choose a GPU if you have multiple per node
    cudaSetDevice(myrank % 4);

    // For demonstration, let’s replicate steps from your original CUDACompress
    // main:

    unsigned int distinctCharacterCount = 0;
    unsigned int combinedHuffmanNodes = 0;
    unsigned char bitSequence[255];
    unsigned char bitSequenceLength = 0;

    // 1) Build Huffman tree on CPU (like your original code does).
    //    If you already do it in MPI code, remove or skip here.

    // Clear global huffmanTreeNode array
    memset(huffmanTreeNode, 0, sizeof(huffmanTreeNode));
    head_huffmanTreeNode = NULL;

    // Count how many distinct characters
    for (unsigned int i = 0; i < 256; i++) {
        if (frequency[i] > 0) {
            huffmanTreeNode[distinctCharacterCount].count = frequency[i];
            huffmanTreeNode[distinctCharacterCount].letter = i;
            huffmanTreeNode[distinctCharacterCount].left = NULL;
            huffmanTreeNode[distinctCharacterCount].right = NULL;
            distinctCharacterCount++;
        }
    }

    // If we have more than 1 distinct symbol, build the full Huffman tree:
    for (unsigned int i = 0; i < distinctCharacterCount - 1; i++) {
        combinedHuffmanNodes = 2 * i;
        sortHuffmanTree(i, distinctCharacterCount, combinedHuffmanNodes);
        buildHuffmanTree(i, distinctCharacterCount, combinedHuffmanNodes);
    }
    // If exactly 1 distinct symbol, special-case:
    if (distinctCharacterCount == 1) {
        head_huffmanTreeNode = &huffmanTreeNode[0];
    }

    // 2) Build Huffman dictionary
    memset(huffmanDictionary, 0, sizeof(huffmanDictionary));
    buildHuffmanDictionary(
        head_huffmanTreeNode,
        bitSequence,
        bitSequenceLength
    );

    // 3) (Optional) In the original code, you do some GPU memory checks
    //    for large input. That’s up to you. For brevity, we skip or keep a
    //    shorter version.

    // 4) We allocate an offset array for the GPU to use for bit positions, etc.
    //    In your original code, you do:
    //       compressedDataOffset = (unsigned int *)malloc((inputFileLength +
    //       1)* ...);
    //    Just replicate as needed:

    unsigned int *compressedDataOffset =
        (unsigned int *)malloc((inputSize + 1) * sizeof(unsigned int));

    // 5) Launch the GPU compression (like your original
    // lauchCUDAHuffmanCompress):
    //    This function presumably does the parallel bit-packing on the GPU.
    //    You had parameters: (inputData, offsetArray, inputSize, numKernelRuns,
    //    integerOverflowFlag, mem_req)
    //
    //    For “numKernelRuns” or “mem_req”, your old code computed them by
    //    looking at the GPU memory, or by some formula. Keep that if you want.
    //    For brevity, we do a simpler approach:

    // Just do a single kernel run by default, or replicate your original logic:
    int numKernelRuns = 1;
    unsigned int integerOverflowFlag = 0; // If your code checks that
    long unsigned int mem_req = 0;        // If your code checks that

    lauchCUDAHuffmanCompress(
        inputFileData,        // pointer to your local chunk
        compressedDataOffset, // we allocated above
        inputSize,            // length
        numKernelRuns,
        integerOverflowFlag,
        mem_req
    );

    // 6) In your original code, the GPU kernel writes the compressed bits into
    // the beginning
    //    of inputFileData or some device buffer. Then you figure out how many
    //    total bits/bytes got used.  In your original `CUDACompress.cu`, you
    //    eventually do something like:
    //         fwrite(&inputFileLength, ...)
    //         fwrite(frequency, ...)
    //         fwrite(inputFileData, 1, mem_offset/8, compressedFile);
    //
    //    Instead of writing to a file here, we want to return the compressed
    //    buffer in
    //    `*hostCompressedData`. So we need to know how many bytes the GPU used.
    //    In your original code, you compute something like `mem_offset/8`.
    //    The code that sets `mem_offset` was:
    //
    //       mem_offset = 0;
    //       for(i = 0; i < 256; i++){
    //          mem_offset += frequency[i] *
    //          huffmanDictionary.bitSequenceLength[i];
    //       }
    //       mem_offset = (mem_offset % 8 == 0) ? mem_offset : (mem_offset + 8 -
    //       mem_offset % 8);
    //
    //    So let’s do that here:

    unsigned long mem_offset = 0;
    for (unsigned int i = 0; i < 256; i++) {
        if (frequency[i] > 0) {
            // # bits for this character = freq[i] * dict.bitSequenceLength[i]
            mem_offset +=
                ((unsigned long)frequency[i] *
                 (unsigned long)huffmanDictionary[i].bitSequenceLength);
        }
    }
    // round up to nearest multiple of 8
    if (mem_offset % 8 != 0) {
        mem_offset += (8 - (mem_offset % 8));
    }

    unsigned int totalCompressedBytes = (unsigned int)(mem_offset / 8);

    // 7) Now we need to allocate a host buffer to hold these compressed bytes
    //    which presumably your kernel wrote to `inputFileData` or some GPU
    //    buffer. In your original code, you wrote them directly from
    //    `inputFileData`. If lauchCUDAHuffmanCompress wrote them into device
    //    memory, you should do a cudaMemcpy back to host.
    //
    //    If your code (like the original) reuses `inputFileData` on the CPU for
    //    the compressed bits, then you can just pass them along.  Let’s assume
    //    the GPU wrote them in-place to `inputFileData`: So we just copy from
    //    that array into a new array of length = totalCompressedBytes.

    *hostCompressedData = (unsigned char *)malloc(totalCompressedBytes);
    memcpy(*hostCompressedData, inputFileData, totalCompressedBytes);

    // 8) Set the final compressed size
    *hostCompressedSize = totalCompressedBytes;

    // 9) Cleanup
    free(compressedDataOffset);

    // (Optional) Debug
    printf(
        "Rank %d (runCudaLand): InputSize=%u compressed to %u bytes\n",
        myrank,
        inputSize,
        totalCompressedBytes
    );
}
