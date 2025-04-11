#include "../../include/parallelHeaderHost.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/** Not all used, but required by our other files */
struct huffmanDictionary huffmanDictionary;
struct huffmanTree *head_huffmanTreeNode;
struct huffmanTree huffmanTreeNode[512];
unsigned char bitSequenceConstMemory[256][255];
unsigned int constMemoryFlag;

void build_hoffman(uint64_t frequency[256]);

int main(int argc, char **argv) {
    clock_t start, end;
    uint64_t cpu_time_used;
    uint64_t outputFileLengthCounter, outputFileLength, frequency[256],
        compressedFileLength;
    unsigned char currentInputBit, currentInputByte, *compressedData,
        *outputData;
    /** The number of chunks in the compressed file */
    uint32_t num_chunks;
    struct huffmanTree *current_huffmanTreeNode;
    FILE *compressedFile, *outputFile;

    // open source compressed file
    compressedFile = fopen(argv[1], "rb");

    // read the header and fill frequency array
    fread(&outputFileLength, sizeof(uint64_t), 1, compressedFile);
    fread(frequency, 256 * sizeof(uint64_t), 1, compressedFile);
    uint64_t sum_global_frequencies = 0;
    for (int i = 0; i < 256; i++) {
        sum_global_frequencies += frequency[i];
    }
    if (sum_global_frequencies != outputFileLength) {
        fprintf(
            stderr,
            "WARNING: sum of global frequencies %llu does not equal output "
            "file length %llu\n",
            sum_global_frequencies,
            outputFileLength
        );
        fclose(compressedFile);
        return EXIT_FAILURE;
    }
    fread(&num_chunks, sizeof(uint32_t), 1, compressedFile);
    uint64_t *chunk_bit_lengths =
        (uint64_t *)malloc(sizeof(uint64_t) * num_chunks);
    fread(chunk_bit_lengths, sizeof(uint64_t), num_chunks, compressedFile);

    printf("File contains %u chunks with bit lengths: [", num_chunks);
    uint64_t expected_compression_len = 0;
    for (uint32_t i = 0; i < num_chunks; i++) {
        if (i != 0) {
            printf(", ");
        }
        printf("%lu", chunk_bit_lengths[i]);
        expected_compression_len +=
            chunk_bit_lengths[i] / 8 + (chunk_bit_lengths[i] % 8 != 0);
    }
    printf("]\n");

    // find length of compressed file
    fseek(compressedFile, 0, SEEK_END);
    compressedFileLength = ftell(compressedFile) - 257 * sizeof(uint64_t) -
                           sizeof(uint32_t) - num_chunks * sizeof(uint64_t);
    if (expected_compression_len != compressedFileLength) {
        fclose(compressedFile);
        fprintf(
            stderr,
            "File length did not match the expected size: expected %lu but "
            "found %lu\n",
            expected_compression_len,
            compressedFileLength
        );
        return EXIT_FAILURE;
    }
    printf("File len matches expected.\n");
    fseek(
        compressedFile,
        257 * sizeof(uint64_t) + sizeof(uint32_t) +
            num_chunks * sizeof(uint64_t),
        SEEK_SET
    );
    printf(
        "Compressed region length %lu, output len %lu\n",
        compressedFileLength,
        outputFileLength
    );

    // allocate required memory and read the file to memoryand then close file
    compressedData = malloc(compressedFileLength * sizeof(unsigned char));
    fread(
        compressedData,
        sizeof(unsigned char),
        compressedFileLength,
        compressedFile
    );
    fclose(compressedFile);
    // start time measure
    start = clock();

    build_hoffman(frequency);

    // write the data to file
    outputData = malloc(outputFileLength * sizeof(unsigned char));
    outputFileLengthCounter = 0;
    uint64_t chunk_start_offset = 0;
    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        current_huffmanTreeNode = head_huffmanTreeNode;

        for (uint64_t i = 0; i < chunk_bit_lengths[chunk]; i++) {
            currentInputByte = compressedData[i / 8 + chunk_start_offset];
            currentInputBit = (currentInputByte >> (7 - i % 8)) & 1;
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

        chunk_start_offset +=
            chunk_bit_lengths[chunk] / 8 + (chunk_bit_lengths[chunk] % 8 != 0);
    }
    // display runtime
    end = clock();
    free(chunk_bit_lengths);

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
    unsigned char bitSequence[255];
    unsigned char bitSequenceLength = 0;

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

    buildHuffmanDictionary(
        head_huffmanTreeNode,
        bitSequence,
        bitSequenceLength
    );
}
