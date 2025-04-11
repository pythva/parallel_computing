/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// Sriram Madhivanan
// Header used for GPU implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#pragma once
#include "parallelHeaderHost.h"
#include <stdint.h>
#include <stdlib.h>

// extern __constant__ unsigned char d_bitSequenceConstMemory[256][255];

__global__ void compress(
    unsigned char *d_inputFileData,
    uint64_t *d_compressedDataOffset,
    struct huffmanDictionary *d_huffmanDictionary,
    unsigned char *d_byteCompressedData,
    uint64_t d_inputFileLength,
    unsigned int constMemoryFlag
);
__global__ void compress(
    unsigned char *d_inputFileData,
    uint64_t *d_compressedDataOffset,
    struct huffmanDictionary *d_huffmanDictionary,
    unsigned char *d_byteCompressedData,
    unsigned char *d_temp_overflow,
    uint64_t d_inputFileLength,
    unsigned int constMemoryFlag,
    uint64_t overflowPosition
);
__global__ void compress(
    unsigned char *d_inputFileData,
    uint64_t *d_compressedDataOffset,
    struct huffmanDictionary *d_huffmanDictionary,
    unsigned char *d_byteCompressedData,
    uint64_t d_lowerPosition,
    unsigned int constMemoryFlag,
    uint64_t d_upperPosition
);
__global__ void compress(
    unsigned char *d_inputFileData,
    uint64_t *d_compressedDataOffset,
    struct huffmanDictionary *d_huffmanDictionary,
    unsigned char *d_byteCompressedData,
    unsigned char *d_temp_overflow,
    uint64_t d_lowerPosition,
    unsigned int constMemoryFlag,
    uint64_t d_upperPosition,
    uint64_t overflowPosition
);

void createDataOffsetArray(
    uint64_t *compressedDataOffset,
    unsigned char *inputFileData,
    uint64_t inputFileLength
);
void createDataOffsetArray(
    uint64_t *compressedDataOffset,
    unsigned char *inputFileData,
    uint64_t inputFileLength,
    uint64_t *gpuMemoryOverflowIndex,
    unsigned int *gpuBitPaddingFlag,
    uint64_t mem_req
);
void createDataOffsetArray(
    uint64_t *compressedDataOffset,
    unsigned char *inputFileData,
    uint64_t inputFileLength,
    uint64_t *integerOverflowIndex,
    unsigned int *bitPaddingFlag,
    uint64_t numBytes
);
void createDataOffsetArray(
    uint64_t *compressedDataOffset,
    unsigned char *inputFileData,
    uint64_t inputFileLength,
    uint64_t *integerOverflowIndex,
    unsigned int *bitPaddingFlag,
    uint64_t *gpuMemoryOverflowIndex,
    unsigned int *gpuBitPaddingFlag,
    uint64_t numBytes,
    uint64_t mem_req
);

void lauchCUDAHuffmanCompress(
    unsigned char *inputFileData,
    uint64_t *compressedDataOffset,
    uint64_t inputFileLength,
    int numKernelRuns,
    unsigned int integerOverflowFlag,
    uint64_t mem_req
);
