/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// Sriram Madhivanan
// Header used for GPU implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#pragma once
#include <stdlib.h>
#include <stdint.h>

struct huffmanDictionary {
    unsigned char bitSequence[256][191];
    unsigned char bitSequenceLength[256];
};

struct huffmanTree {
    unsigned char letter;
    uint64_t count;
    struct huffmanTree *left, *right;
};

extern struct huffmanTree *head_huffmanTreeNode;
extern struct huffmanTree huffmanTreeNode[512];
extern unsigned char bitSequenceConstMemory[256][255];
extern unsigned int constMemoryFlag;
extern struct huffmanDictionary huffmanDictionary;
// extern __constant__ unsigned char d_bitSequenceConstMemory[256][255];

void sortHuffmanTree(
    int i,
    int distinctCharacterCount,
    int combinedHuffmanNodes
);
void buildHuffmanTree(
    int i,
    int distinctCharacterCount,
    int combinedHuffmanNodes
);
void buildHuffmanDictionary(
    struct huffmanTree *root,
    unsigned char *bitSequence,
    unsigned char bitSequenceLength
);

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
