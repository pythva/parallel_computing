/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// Sriram Madhivanan
// Functions used for GPU and CUDA-MPI implementations
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

#include "parallelHeader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// generate data offset array
// case - single run, no overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void createDataOffsetArray(
    uint64_t *compressedDataOffset,
    unsigned char *inputFileData,
    uint64_t inputFileLength
) {
    uint64_t i;
    compressedDataOffset[0] = 0;
    for (i = 0; i < inputFileLength; i++) {
        compressedDataOffset[i + 1] =
            huffmanDictionary.bitSequenceLength[inputFileData[i]] +
            compressedDataOffset[i];
    }
    if (compressedDataOffset[inputFileLength] % 8 != 0) {
        compressedDataOffset[inputFileLength] =
            compressedDataOffset[inputFileLength] +
            (8 - (compressedDataOffset[inputFileLength] % 8));
    }
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// generate data offset array
// case - single run, with overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void createDataOffsetArray(
    uint64_t *compressedDataOffset,
    unsigned char *inputFileData,
    uint64_t inputFileLength,
    uint64_t *integerOverflowIndex,
    unsigned int *bitPaddingFlag,
    int numBytes
) {
    int i, j;
    // calculate compressed data offset - (1048576 is a safe number that will
    // ensure there is no integer overflow in GPU, it should be minimum 8 *
    // number of threads)
    j = 0;
    compressedDataOffset[0] = 0;
    for (i = 0; i < inputFileLength; i++) {
        compressedDataOffset[i + 1] =
            huffmanDictionary.bitSequenceLength[inputFileData[i]] +
            compressedDataOffset[i];
        if (compressedDataOffset[i + 1] + numBytes < compressedDataOffset[i]) {
            integerOverflowIndex[j] = i;
            if (compressedDataOffset[i] % 8 != 0) {
                bitPaddingFlag[j] = 1;
                compressedDataOffset[i + 1] =
                    (compressedDataOffset[i] % 8) +
                    huffmanDictionary.bitSequenceLength[inputFileData[i]];
                compressedDataOffset[i] = compressedDataOffset[i] +
                                          (8 - (compressedDataOffset[i] % 8));
            } else {
                compressedDataOffset[i + 1] =
                    huffmanDictionary.bitSequenceLength[inputFileData[i]];
            }
            j++;
        }
    }
    if (compressedDataOffset[inputFileLength] % 8 != 0) {
        compressedDataOffset[inputFileLength] =
            compressedDataOffset[inputFileLength] +
            (8 - (compressedDataOffset[inputFileLength] % 8));
    }
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// generate data offset array
// case - multiple run, no overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void createDataOffsetArray(
    uint64_t *compressedDataOffset,
    unsigned char *inputFileData,
    uint64_t inputFileLength,
    uint64_t *gpuMemoryOverflowIndex,
    unsigned int *gpuBitPaddingFlag,
    uint64_t mem_req
) {
    int i, j;
    j = 0;
    gpuMemoryOverflowIndex[0] = 0;
    gpuBitPaddingFlag[0] = 0;
    compressedDataOffset[0] = 0;
    for (i = 0; i < inputFileLength; i++) {
        compressedDataOffset[i + 1] =
            huffmanDictionary.bitSequenceLength[inputFileData[i]] +
            compressedDataOffset[i];
        if (compressedDataOffset[i + 1] > mem_req) {
            gpuMemoryOverflowIndex[j * 2 + 1] = i;
            gpuMemoryOverflowIndex[j * 2 + 2] = i + 1;
            if (compressedDataOffset[i] % 8 != 0) {
                gpuBitPaddingFlag[j + 1] = 1;
                compressedDataOffset[i + 1] =
                    (compressedDataOffset[i] % 8) +
                    huffmanDictionary.bitSequenceLength[inputFileData[i]];
                compressedDataOffset[i] = compressedDataOffset[i] +
                                          (8 - (compressedDataOffset[i] % 8));
            } else {
                compressedDataOffset[i + 1] =
                    huffmanDictionary.bitSequenceLength[inputFileData[i]];
            }
            j++;
        }
    }
    if (compressedDataOffset[inputFileLength] % 8 != 0) {
        compressedDataOffset[inputFileLength] =
            compressedDataOffset[inputFileLength] +
            (8 - (compressedDataOffset[inputFileLength] % 8));
    }
    gpuMemoryOverflowIndex[j * 2 + 1] = inputFileLength;
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// generate data offset array
// case - multiple run, with overflow
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
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
) {
    int i, j, k;
    j = 0;
    k = 0;
    compressedDataOffset[0] = 0;
    for (i = 0; i < inputFileLength; i++) {
        compressedDataOffset[i + 1] =
            huffmanDictionary.bitSequenceLength[inputFileData[i]] +
            compressedDataOffset[i];
        if (j != 0 && (compressedDataOffset[i + 1] +
                           compressedDataOffset[integerOverflowIndex[j - 1]] >
                       mem_req)) {
            gpuMemoryOverflowIndex[k * 2 + 1] = i;
            gpuMemoryOverflowIndex[k * 2 + 2] = i + 1;
            if (compressedDataOffset[i] % 8 != 0) {
                gpuBitPaddingFlag[k + 1] = 1;
                compressedDataOffset[i + 1] =
                    (compressedDataOffset[i] % 8) +
                    huffmanDictionary.bitSequenceLength[inputFileData[i]];
                compressedDataOffset[i] = compressedDataOffset[i] +
                                          (8 - (compressedDataOffset[i] % 8));
            } else {
                compressedDataOffset[i + 1] =
                    huffmanDictionary.bitSequenceLength[inputFileData[i]];
            }
            k++;
        } else if (compressedDataOffset[i + 1] + numBytes <
                   compressedDataOffset[i]) {
            integerOverflowIndex[j] = i;
            if (compressedDataOffset[i] % 8 != 0) {
                bitPaddingFlag[j] = 1;
                compressedDataOffset[i + 1] =
                    (compressedDataOffset[i] % 8) +
                    huffmanDictionary.bitSequenceLength[inputFileData[i]];
                compressedDataOffset[i] = compressedDataOffset[i] +
                                          (8 - (compressedDataOffset[i] % 8));
            } else {
                compressedDataOffset[i + 1] =
                    huffmanDictionary.bitSequenceLength[inputFileData[i]];
            }
            j++;
        }
    }
    if (compressedDataOffset[inputFileLength] % 8 != 0) {
        compressedDataOffset[inputFileLength] =
            compressedDataOffset[inputFileLength] +
            (8 - (compressedDataOffset[inputFileLength] % 8));
    }
    gpuMemoryOverflowIndex[j * 2 + 1] = inputFileLength;
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*---------------------------------------------------------------------------------------------------------------------------------------------*/