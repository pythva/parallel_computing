/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// Sriram Madhivanan
// Header used for serial and MPI-only implementations
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#pragma once
#include <stdint.h>
#include <stdlib.h>

struct huffmanDictionary {
    unsigned char bitSequence[255];
    unsigned char bitSequenceLength;
};

struct huffmanTree {
    unsigned char letter;
    uint64_t count;
    struct huffmanTree *left, *right;
};

extern struct huffmanDictionary huffmanDictionary[256];
extern struct huffmanTree *head_huffmanTreeNode;
extern struct huffmanTree huffmanTreeNode[512];

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
