/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// Sriram Madhivanan
// MPI Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/

#include "../../include/clockcycle.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include "../include/serialHeader.h"
// extern void runCudaLand( int myrank );
// struct huffmanDictionary huffmanDictionary[256];
// struct huffmanTree *head_huffmanTreeNode = NULL;
// struct huffmanTree huffmanTreeNode[512];

extern void runCudaLand(
    int myrank,
    unsigned char *inputData, // local chunk
    uint64_t blockLength,
    uint64_t *frequency,              // global freq
    unsigned char **d_compressedData, // returned GPU comp buffer
    uint64_t *compBlockLength
); // returned comp length

int main(int argc, char *argv[]) {
    uint64_t start, end;
    // unsigned int cpu_time_used;
    unsigned int rank, numProcesses;
    uint64_t i, j, blockLength;
    uint64_t *compBlockLengthArray;
    // unsigned int distinctCharacterCount, combinedHuffmanNodes;
    uint64_t frequency[256] = {0}, inputFileLength, compBlockLength;
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
        MPI_IN_PLACE,
        frequency,
        256,
        MPI_UINT64_T,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    // compressedData = (unsigned char *)malloc(blockLength * sizeof(unsigned
    // char));
    compBlockLengthArray = (uint64_t *)malloc(numProcesses * sizeof(uint64_t));

    /*
    // initialize nodes of huffman tree
    distinctCharacterCount = 0;
    for (i = 0; i < 256; i++){
        if (frequency[i] > 0){
            huffmanTreeNode[distinctCharacterCount].count = frequency[i];
            huffmanTreeNode[distinctCharacterCount].letter = i;
            huffmanTreeNode[distinctCharacterCount].left = NULL;
            huffmanTreeNode[distinctCharacterCount].right = NULL;
            distinctCharacterCount++;
        }
    }

    // build tree
    for (i = 0; i < distinctCharacterCount - 1; i++){
        combinedHuffmanNodes = 2 * i;
        sortHuffmanTree(i, distinctCharacterCount, combinedHuffmanNodes);
        buildHuffmanTree(i, distinctCharacterCount, combinedHuffmanNodes);
    }

    if(distinctCharacterCount == 1){
      head_huffmanTreeNode = &huffmanTreeNode[0];
    }

    // build table having the bitSequence sequence and its length
    buildHuffmanDictionary(head_huffmanTreeNode, bitSequence,
    bitSequenceLength);
    // compress
    compBlockLength = 0;
    for (i = 0; i < blockLength; i++){
        for (j = 0; j < huffmanDictionary[inputFileData[i]].bitSequenceLength;
    j++){ if (huffmanDictionary[inputFileData[i]].bitSequence[j] == 0){ writeBit
    = writeBit << 1; bitsFilled++;
            }
            else{
                writeBit = (writeBit << 1) | 01;
                bitsFilled++;
            }
            if (bitsFilled == 8){
                compressedData[compBlockLength] = writeBit;
                bitsFilled = 0;
                writeBit = 0;
                compBlockLength++;
            }
        }
    }

    if (bitsFilled != 0){
        for (i = 0; (unsigned char)i < 8 - bitsFilled; i++){
            writeBit = writeBit << 1;
        }
        compressedData[compBlockLength] = writeBit;
        compBlockLength++;
    }

    // calculate length of compressed data
    //compBlockLength = compBlockLength + 1024;
    */

    // unsigned char *gpuCompressed = NULL; // the bridging function will
    // allocate
    //  and return a pointer. Or we can do it ourselves.
    // unsigned int gpuCompSize = 0;

    unsigned char *gpuCompressedData = NULL;
    uint64_t gpuCompressedSize = 0;

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
    for (int i = 0; i < 256; i++) {
        if (frequency[i] > 0) {
            printf("%c: %llu  ", (i >= 32 && i <= 126 ? i : '.'), frequency[i]);
        }
    }
    runCudaLand(
        rank,
        inputFileData,
        blockLength,
        frequency,
        &gpuCompressedData,
        &gpuCompressedSize
    );

    compressedData = NULL;               // init first
    compressedData = gpuCompressedData;  // pass the data
    compBlockLength = gpuCompressedSize; // pass the data size
    compBlockLengthArray[rank] = compBlockLength;

    printf("the size of the data%d", compBlockLength);
    MPI_Gather(
        &compBlockLength,
        1,
        MPI_UINT64_T,
        compBlockLengthArray,
        1,
        MPI_UINT64_T,
        0,
        MPI_COMM_WORLD
    );

    // ---- RANK 0 COMPUTES OFFSETS ----
    // We’ll store the starting offset for each rank’s data in
    // compBlockOffset[].
    uint64_t *compBlockOffsetArray = NULL; // NEW
    uint64_t currentOffset;
    if (rank == 0) {
        compBlockOffsetArray =
            (uint64_t *)malloc(numProcesses * sizeof(uint64_t));
        currentOffset = sizeof(uint64_t) /*for inputFileLength*/ +
                        sizeof(uint64_t) * 256 /*for frequency[256]*/;

        // Rank 0 starts at offset = currentOffset
        compBlockOffsetArray[0] = currentOffset;

        // Then each subsequent rank i starts after rank (i-1) data
        for (int i = 1; i < numProcesses; i++) {
            currentOffset += compBlockLengthArray[i - 1];
            compBlockOffsetArray[i] = currentOffset;
        }
    }
    if (rank == 0) {
        // Rank 0 has computed compBlockOffsetArray
    } else {
        compBlockOffsetArray =
            (uint64_t *)malloc(numProcesses * sizeof(uint64_t));
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
            sizeof(uint64_t),
            frequency,
            256,
            MPI_UINT64_T,
            MPI_STATUS_IGNORE
        );
        // If you want to store `numProcesses` or anything else in the header,
        // write it now in the appropriate offsets.
    }

    MPI_File_write_at(
        mpi_compressedFile,
        compBlockOffsetArray[rank],
        compressedData,
        compBlockLength, // number of bytes from this rank
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
            currentOffset,
            end - start
        );
    }

    if (compBlockOffsetArray != NULL) {
        free(compBlockOffsetArray);
    }

    free(inputFileData);
    free(compressedData);
    // if (compressedData) free(compressedData);
    MPI_Finalize();
    return 0;
}
