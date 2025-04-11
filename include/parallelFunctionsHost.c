#include "parallelHeaderHost.h"
#include <stdlib.h>


/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// sortHuffmanTree nodes based on frequency
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void sortHuffmanTree(
    int i,
    int distinctCharacterCount,
    int combinedHuffmanNodes
) {
    int a, b;
    for (a = combinedHuffmanNodes; a < distinctCharacterCount - 1 + i; a++) {
        for (b = combinedHuffmanNodes; b < distinctCharacterCount - 1 + i;
             b++) {
            if (huffmanTreeNode[b].count > huffmanTreeNode[b + 1].count) {
                struct huffmanTree temp_huffmanTreeNode = huffmanTreeNode[b];
                huffmanTreeNode[b] = huffmanTreeNode[b + 1];
                huffmanTreeNode[b + 1] = temp_huffmanTreeNode;
            }
        }
    }
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// build tree based on sortHuffmanTree result
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void buildHuffmanTree(
    int i,
    int distinctCharacterCount,
    int combinedHuffmanNodes
) {
    huffmanTreeNode[distinctCharacterCount + i].count =
        huffmanTreeNode[combinedHuffmanNodes].count +
        huffmanTreeNode[combinedHuffmanNodes + 1].count;
    huffmanTreeNode[distinctCharacterCount + i].left =
        &huffmanTreeNode[combinedHuffmanNodes];
    huffmanTreeNode[distinctCharacterCount + i].right =
        &huffmanTreeNode[combinedHuffmanNodes + 1];
    head_huffmanTreeNode = &(huffmanTreeNode[distinctCharacterCount + i]);
}
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
// get bitSequence sequence for each char value
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
void buildHuffmanDictionary(
    struct huffmanTree *root,
    unsigned char *bitSequence,
    unsigned char bitSequenceLength
) {
    if (root->left) {
        bitSequence[bitSequenceLength] = 0;
        buildHuffmanDictionary(root->left, bitSequence, bitSequenceLength + 1);
    }

    if (root->right) {
        bitSequence[bitSequenceLength] = 1;
        buildHuffmanDictionary(root->right, bitSequence, bitSequenceLength + 1);
    }

    if (root->left == NULL && root->right == NULL) {
        huffmanDictionary.bitSequenceLength[root->letter] = bitSequenceLength;
        if (bitSequenceLength < 192) {
            memcpy(
                huffmanDictionary.bitSequence[root->letter],
                bitSequence,
                bitSequenceLength * sizeof(unsigned char)
            );
        } else {
            memcpy(
                bitSequenceConstMemory[root->letter],
                bitSequence,
                bitSequenceLength * sizeof(unsigned char)
            );
            memcpy(
                huffmanDictionary.bitSequence[root->letter],
                bitSequence,
                191
            );
            constMemoryFlag = 1;
        }
    }
}