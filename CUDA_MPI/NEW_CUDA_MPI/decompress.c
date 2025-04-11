#include "../../include/serialHeader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/** Not all used, but required by our other files */
struct huffmanDictionary huffmanDictionary[256];
struct huffmanTree *head_huffmanTreeNode;
struct huffmanTree huffmanTreeNode[512];

void build_hoffman(uint64_t frequency[256]);

int main(int argc, char **argv) {
    clock_t start, end;
    uint64_t cpu_time_used;
    uint64_t i, j;
    uint64_t outputFileLengthCounter, outputFileLength, frequency[256],
        compressedFileLength;
    unsigned char currentInputBit, currentInputByte, *compressedData,
        *outputData;
    struct huffmanTree *current_huffmanTreeNode;
    FILE *compressedFile, *outputFile;

    // open source compressed file
    compressedFile = fopen(argv[1], "rb");

    // read the header and fill frequency array
    fread(&outputFileLength, sizeof(uint64_t), 1, compressedFile);
    fread(frequency, 256 * sizeof(uint64_t), 1, compressedFile);

    // find length of compressed file
    fseek(compressedFile, 0, SEEK_END);
    compressedFileLength = ftell(compressedFile) - 257 * sizeof(uint64_t);
    fseek(compressedFile, 257 * sizeof(uint64_t), SEEK_SET);

    // allocate required memory and read the file to memoryand then close file
    compressedData = malloc((compressedFileLength) * sizeof(unsigned char));
    fread(
        compressedData,
        sizeof(unsigned char),
        (compressedFileLength),
        compressedFile
    );
    fclose(compressedFile);

    // start time measure
    start = clock();

    build_hoffman(frequency);

    // write the data to file
    outputData = malloc(outputFileLength * sizeof(unsigned char));
    current_huffmanTreeNode = head_huffmanTreeNode;
    outputFileLengthCounter = 0;
    for (i = 0; i < compressedFileLength; i++) {
        currentInputByte = compressedData[i];
        for (j = 0; j < 8; j++) {
            currentInputBit = currentInputByte & 0200;
            currentInputByte = currentInputByte << 1;
            if (currentInputBit == 0) {
                current_huffmanTreeNode = current_huffmanTreeNode->left;
                if (current_huffmanTreeNode->left == NULL) {
                    outputData[outputFileLengthCounter] =
                        current_huffmanTreeNode->letter;
                    current_huffmanTreeNode = head_huffmanTreeNode;
                    outputFileLengthCounter++;
                }
            } else {
                current_huffmanTreeNode = current_huffmanTreeNode->right;
                if (current_huffmanTreeNode->right == NULL) {
                    outputData[outputFileLengthCounter] =
                        current_huffmanTreeNode->letter;
                    current_huffmanTreeNode = head_huffmanTreeNode;
                    outputFileLengthCounter++;
                }
            }
        }
    }

    // display runtime
    end = clock();

    // write decompressed file
    outputFile = fopen(argv[2], "wb");
    fwrite(outputData, sizeof(unsigned char), outputFileLength, outputFile);
    fclose(outputFile);

    cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
    printf("Time taken: %d:%d s\n", cpu_time_used / 1000, cpu_time_used % 1000);
    free(outputData);
    free(compressedData);
    return 0;
}

/**
 * @brief Given global frequencies, sets up the global hoffman tree
 *
 * @param frequency The set of frequencies for each byte
 */
void build_hoffman(uint64_t frequency[256]) {
    uint64_t i;
    unsigned int distinctCharacterCount = 0, combinedHuffmanNodes = 0;

    head_huffmanTreeNode = NULL;
    memset(huffmanTreeNode, 0, sizeof(huffmanTreeNode));

    // initialize nodes of huffman tree
    distinctCharacterCount = 0;
    for (i = 0; i < 256; i++) {
        if (frequency[i] > 0) {
            huffmanTreeNode[distinctCharacterCount].count = frequency[i];
            huffmanTreeNode[distinctCharacterCount].letter = i;
            huffmanTreeNode[distinctCharacterCount].left = NULL;
            huffmanTreeNode[distinctCharacterCount].right = NULL;
            distinctCharacterCount++;
        }
    }

    // build tree
    for (i = 0; i < distinctCharacterCount - 1; i++) {
        combinedHuffmanNodes = 2 * i;
        sortHuffmanTree(i, distinctCharacterCount, combinedHuffmanNodes);
        buildHuffmanTree(i, distinctCharacterCount, combinedHuffmanNodes);
    }
}
