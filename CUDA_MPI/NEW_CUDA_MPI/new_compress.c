// Based on Sriram Madhivanan's MPI implementation

#include "../../include/clockcycle.h"
#include "../../include/parallelHeaderHost.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct huffmanTree *head_huffmanTreeNode;
struct huffmanTree huffmanTreeNode[512];
struct huffmanDictionary huffmanDictionary;
unsigned char bitSequenceConstMemory[256][255];
unsigned int constMemoryFlag = 0;

extern void runCudaLand(
    int myrank,
    unsigned char *inputFileData,
    uint64_t inputFileLength,
    unsigned char **hostCompressedData,
    uint64_t hostCompressedSize
);

void build_hoffman(uint64_t frequency[256]);

int main(int argc, char *argv[]) {
    uint64_t start, end;
    // unsigned int cpu_time_used;
    unsigned int rank, numProcesses;
    uint64_t i, j, blockLength;
    uint64_t *bit_length_array;
    // unsigned int distinctCharacterCount, combinedHuffmanNodes;
    uint64_t frequency[256] = {0}, global_frequency[256], inputFileLength;
    unsigned char *inputFileData, *compressedData;
    // , writeBit = 0, bitsFilled = 0, bitSequence[255], bitSequenceLength = 0;
    FILE *inputFile;

    MPI_Init(&argc, &argv);
    MPI_File mpi_inputFile, mpi_compressedFile;
    MPI_Status status;

    // get rank and number of processes value
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        start = clock_now();
    }

    // get file size
    if (rank == 0) {
        inputFile = fopen(argv[1], "rb");
        fseek(inputFile, 0, SEEK_END);
        inputFileLength = ftell(inputFile);
        fseek(inputFile, 0, SEEK_SET);
        fclose(inputFile);
    }

    // broadcast size of file to all the processes
    MPI_Bcast(&inputFileLength, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    // get file chunk size
    blockLength = inputFileLength / numProcesses;

    if (rank == (numProcesses - 1)) {
        blockLength = inputFileLength - ((numProcesses - 1) * blockLength);
    }

    // open file in each process and read data and allocate memory for
    // compressed data
    MPI_File_open(
        MPI_COMM_WORLD,
        argv[1],
        MPI_MODE_RDONLY,
        MPI_INFO_NULL,
        &mpi_inputFile
    );
    MPI_File_seek(mpi_inputFile, rank * blockLength, MPI_SEEK_SET);

    inputFileData =
        (unsigned char *)malloc(blockLength * sizeof(unsigned char));
    MPI_File_read(
        mpi_inputFile,
        inputFileData,
        blockLength,
        MPI_UNSIGNED_CHAR,
        &status
    );


    // find the frequency of each symbols
    for (i = 0; i < blockLength; i++) {
        frequency[inputFileData[i]]++;
    }

    // a mpi allreduce so all the rank will use the same hoffman tree
    MPI_Allreduce(
        frequency,
        global_frequency,
        256,
        MPI_UINT64_T,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    build_hoffman(global_frequency);

    /** The expected length in bits of this chunk */
    uint64_t chunk_bit_length = 0;
    for (i = 0; i < 256; ++i) {
        chunk_bit_length +=
            huffmanDictionary.bitSequenceLength[i] * frequency[i];
    }

    // printf("%d", blockLength);
    printf("Rank %d: blockLength = %d\n", rank, blockLength);

    // Print inputFileData content (first N bytes for brevity)
    printf("Rank %d: inputFileData = [", rank);
    for (uint64_t i = 0; i < (blockLength < 8 ? blockLength : 8); i++) {
        printf("%02x ", inputFileData[i]); // Print in hex for readability
    }
    printf("...");
    for (uint64_t i = (blockLength < 16 ? 0 : blockLength - 16);
         i < blockLength;
         i++) {
        printf(" %02x", inputFileData[i]); // Print in hex for readability
    }
    printf("]\n");

    printf("Rank %d: frequency[] = { ", rank);
    uint64_t sum_global_frequencies = 0;
    for (int i = 0; i < 256; i++) {
        sum_global_frequencies += global_frequency[i];
        if (global_frequency[i] > 0) {
            printf(
                "%c: %llu  ",
                (i >= 32 && i <= 126 ? i : '.'),
                global_frequency[i]
            );
        }
    }
    printf("}\n");
    if (sum_global_frequencies != inputFileLength) {
        fprintf(
            stderr,
            "WARNING: sum of global frequencies %llu does not equal input file "
            "length %llu\n",
            sum_global_frequencies,
            inputFileLength
        );
    }

    runCudaLand(
        rank,
        inputFileData,
        blockLength,
        &compressedData,
        chunk_bit_length
    );

    bit_length_array = (uint64_t *)malloc(numProcesses * sizeof(uint64_t));
    bit_length_array[rank] = chunk_bit_length;

    printf("the size of the data %d bits\n", chunk_bit_length);
    MPI_Gather(
        &chunk_bit_length,
        1,
        MPI_UINT64_T,
        bit_length_array,
        1,
        MPI_UINT64_T,
        0,
        MPI_COMM_WORLD
    );

    // ---- RANK 0 COMPUTES OFFSETS ----
    // We’ll store the starting offset for each rank’s data in
    // compBlockOffset[].
    uint64_t *compBlockOffsetArray =
        (uint64_t *)malloc(numProcesses * sizeof(uint64_t));
    if (rank == 0) {
        uint64_t currentOffset =
            sizeof(uint64_t) /*for inputFileLength*/ +
            sizeof(uint64_t) * 256 /*for frequency[256]*/ +
            sizeof(uint32_t) /*for number of chunks*/ +
            sizeof(uint64_t) * numProcesses /*for chunk bit lengths*/;

        printf("With %lu chunks and bit lengths [", numProcesses);
        for (int i = 0; i < numProcesses; i++) {
            compBlockOffsetArray[i] = currentOffset;
            currentOffset +=
                bit_length_array[i] / 8 + (bit_length_array[i] % 8 != 0);
            if (i != 0) {
                printf(", ");
            }
            printf("%llu", bit_length_array[i]);
        }
        printf("]\n");
    }
    MPI_Bcast(
        compBlockOffsetArray,
        numProcesses,
        MPI_UINT64_T,
        0,
        MPI_COMM_WORLD
    );

    // ---- OPEN OUTPUT FILE ----
    MPI_File_open(
        MPI_COMM_WORLD,
        argv[2],
        MPI_MODE_CREATE | MPI_MODE_WRONLY,
        MPI_INFO_NULL,
        &mpi_compressedFile
    );

    if (rank == 0) {
        // Write the inputFileLength at offset 0
        MPI_File_write_at(
            mpi_compressedFile,
            0,
            &inputFileLength,
            1,
            MPI_UINT64_T,
            MPI_STATUS_IGNORE
        );

        // Write frequency array at offset 8
        MPI_File_write_at(
            mpi_compressedFile,
            8,
            global_frequency,
            256,
            MPI_UINT64_T,
            MPI_STATUS_IGNORE
        );

        // Write numProcesses at offset 16
        MPI_File_write_at(
            mpi_compressedFile,
            8 * 257,
            &numProcesses,
            1,
            MPI_UINT32_T,
            MPI_STATUS_IGNORE
        );

        // Write each chunk's bit length
        MPI_File_write_at(
            mpi_compressedFile,
            8 * 257 + 4,
            bit_length_array,
            numProcesses,
            MPI_UINT64_T,
            MPI_STATUS_IGNORE
        );
    }

    MPI_File_write_at(
        mpi_compressedFile,
        compBlockOffsetArray[rank],
        compressedData,
        chunk_bit_length / 8 + (chunk_bit_length % 8 != 0),
        MPI_UNSIGNED_CHAR,
        MPI_STATUS_IGNORE
    );

    MPI_File_close(&mpi_compressedFile);
    MPI_File_close(&mpi_inputFile);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        end = clock_now();
        printf(
            "Finished with %u ranks | input size %llu | output size %llu | "
            "nanoseconds %llu\n",
            numProcesses,
            inputFileLength,
            chunk_bit_length / 8 + (chunk_bit_length % 8 != 0),
            end - start
        );
    }

    free(compBlockOffsetArray);
    free(inputFileData);
    free(compressedData);

    MPI_Finalize();
    return 0;
}


/**
 * @brief Given global frequencies, sets up the global hoffman dictionary
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
    buildHuffmanDictionary(
        head_huffmanTreeNode,
        bitSequence,
        bitSequenceLength
    );
}
