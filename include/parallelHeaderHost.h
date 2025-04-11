#pragma once
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
extern struct huffmanDictionary huffmanDictionary;
extern unsigned char bitSequenceConstMemory[256][255];
extern unsigned int constMemoryFlag;

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
