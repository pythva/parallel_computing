#include "../../include/parallelHeaderHost.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/// This is a silly thing the previous implementation did
/// I think it's to reduce shared memory?
__constant__ unsigned char d_bitSequenceConstMemory[256][255];


// The number of symbols each thread will handle
#define GROUP_LEN 256

#define MAX_GPU_INPUT (4UL * 1024UL * 1024UL * 1024UL)

extern "C" {
void launch_compression(
    int rank,
    unsigned char *full_input,
    uint64_t full_input_len,
    unsigned char *output,
    uint64_t output_len
);
}

#define DEBUG_CPU
#ifdef DEBUG_CPU
void calculate_len_cpu(
    unsigned char *input,
    uint64_t input_len,
    struct huffmanDictionary *dictionary,
    uint64_t *len
) {
    for (uint64_t i = 0; i * GROUP_LEN < input_len; i++) {
        uint64_t my_len = 0;
        for (uint64_t j = i * GROUP_LEN;
             j < i * GROUP_LEN + GROUP_LEN && j < input_len;
             j++) {
            my_len += dictionary->bitSequenceLength[input[j]];
        }
        len[i] = my_len;
    }
}
#endif

/**
 * @brief Calculates the combined length of each group of symbols.
 *
 * This is achieved by summing the bit lengths of the huffman codes for each
 * byte in the group.
 *
 * TODO: this could probably be better optimized for memory accesses
 *
 * @param input The uncompressed input data
 * @param input_len The length in bytes of the input data
 * @param dictionary The huffman dictionary to use in calculations
 * @param len [out] in bits for each thread
 */
__global__ void calculate_len(
    unsigned char *input,
    uint64_t input_len,
    struct huffmanDictionary *dictionary,
    uint64_t *len
) {
    __shared__ struct huffmanDictionary table;
    if (threadIdx.x == 0) {
        // TODO: this might be parallelizable
        memcpy(&table, dictionary, sizeof(struct huffmanDictionary));
    }

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __syncthreads();

    // Each thread can do multiple groups of symbols
    for (uint64_t i = tid; i * GROUP_LEN < input_len;
         i += gridDim.x * blockDim.x) {
        // For each group of symbols
        uint64_t my_len = 0;
        for (uint64_t j = i * GROUP_LEN;
             j < i * GROUP_LEN + GROUP_LEN && j < input_len;
             j++) {
            my_len += table.bitSequenceLength[input[j]];
        }
        len[i] = my_len;
    }
}


/**
 * @brief Calculates offsets from the compressed length in bits of each group's
 * data.
 *
 * TODO: this can be made more efficient by doing parallelizing with a tree-like
 * partial reduction.
 *
 * @param array [in/out] Input is the compressed length of each group, output is
 * their offsets (both in bits)
 * @param group_count The number of groups, which is the length of the array
 * @param full_len [out] the full length in bits (sum of all lengths + initial
 * offset)
 * @param initial_offset is the initial offset, for use when we can't fit a
 * rank's whole data on the GPU
 */
void calculate_offsets(
    uint64_t *array,
    uint64_t group_count,
    uint64_t *full_len,
    uint64_t initial_offset
) {
    uint64_t curr = initial_offset;
    for (uint64_t i = 0; i < group_count; i++) {
        uint64_t len = array[i];
        array[i] = curr;
        curr += len;
    }
    *full_len = curr;
}

#ifdef DEBUG_CPU
void apply_compression_cpu(
    unsigned char *input,
    uint64_t input_len,
    uint64_t *offsets,
    unsigned char *output
) {
    for (uint64_t i = 0; i * GROUP_LEN < input_len; i++) {
        uint64_t out_pos = offsets[i];
        for (uint64_t in_pos = i * GROUP_LEN;
             in_pos < i * GROUP_LEN + GROUP_LEN && in_pos < input_len;
             in_pos++) {
            for (uint64_t k = 0;
                 k < huffmanDictionary.bitSequenceLength[input[in_pos]];
                 k++) {
                unsigned char bit;
                // See above regarding the const memory thing. Probably should
                // change this.
                if (k < 191) {
                    bit = huffmanDictionary.bitSequence[input[in_pos]][k];
                } else {
                    bit = bitSequenceConstMemory[input[in_pos]][k];
                }
                output[out_pos / 8] |= bit << (7 - out_pos % 8);
                out_pos++;
            }
        }
    }
}
#endif

/**
 * @brief Applies the huffman coding to the input array, using pre-calculated
 * locations (bit offsets) that each group of `GROUP_LEN` input bytes will start
 * at when compressed. Should be called twice for each input, once with each
 * parity, in order to avoid race conditions.
 *
 * @param input The input text loaded onto the GPU
 * @param input_len The length of the input text
 * @param dictionary The huffman code dictionary
 * @param offsets The bit offsets into `output` that each group of
 * `GROUP_LEN` bytes will start at (calculated by `calculate_len`)
 * @param output [out] Should be filled with zeroes and have enough space for
 * the compressed data
 * @param parity To avoid race conditions in words shared between two groups, we
 * only do even (0) or odd (1) groups in a single kernel call. Since the CUDA
 * cores do more than one group each for most mid-size or large files, this
 * shouldn't hurt performance much.
 */
__global__ void apply_compression(
    unsigned char *input,
    uint64_t input_len,
    struct huffmanDictionary *dictionary,
    uint64_t *offsets,
    unsigned char *output,
    uint64_t parity
) {
    __shared__ struct huffmanDictionary table;
    if (threadIdx.x == 0) {
        // TODO: this might be parallelizable
        memcpy(&table, dictionary, sizeof(struct huffmanDictionary));
    }

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    parity = parity & 1;
    __syncthreads();

    // Each thread can do multiple groups of symbols
    for (uint64_t i = tid * 2 + parity; i * GROUP_LEN < input_len;
         i += gridDim.x * blockDim.x * 2) {
        // For each group:
        uint64_t out_pos = offsets[i];
        for (uint64_t in_pos = i * GROUP_LEN;
             in_pos < i * GROUP_LEN + GROUP_LEN && in_pos < input_len;
             in_pos++) {
            // TODO: potential optimization by changing table encoding away from
            // one bit per byte
            for (uint64_t k = 0; k < table.bitSequenceLength[input[in_pos]];
                 k++) {
                unsigned char bit;
                // See above regarding the const memory thing. Probably should
                // change this.
                if (k < 191) {
                    bit = table.bitSequence[input[in_pos]][k];
                } else {
                    bit = d_bitSequenceConstMemory[input[in_pos]][k];
                }
                output[out_pos / 8] |= bit << (7 - out_pos % 8);
                out_pos++;
            }
        }
    }
}

void launch_compression(
    int rank,
    unsigned char *full_input,
    uint64_t full_input_len,
    unsigned char *output,
    uint64_t output_len
) {
    cudaSetDevice(rank % 4);

    if (constMemoryFlag == 1) {
        cudaMemcpyToSymbol(
            d_bitSequenceConstMemory,
            bitSequenceConstMemory,
            256 * 255 * sizeof(unsigned char)
        );
        printf("Had to do bitsequence mem\n");
    }

    uint64_t offset = 0;
    unsigned char *d_input;
    cudaMallocManaged(&d_input, min(MAX_GPU_INPUT, full_input_len));
    uint64_t *m_groups; // somewhat over-allocated but that's ok
    cudaMallocManaged(&m_groups, sizeof(uint64_t) * MAX_GPU_INPUT / GROUP_LEN);
    struct huffmanDictionary *d_dictionary;
    cudaMalloc(&d_dictionary, sizeof(struct huffmanDictionary));
    cudaMemcpy(
        d_dictionary,
        &huffmanDictionary,
        sizeof(struct huffmanDictionary),
        cudaMemcpyHostToDevice
    );
    printf("Setup\n");
    for (uint64_t input_pos = 0; input_pos < full_input_len;
         input_pos += MAX_GPU_INPUT) {
        uint64_t input_len = min(MAX_GPU_INPUT, full_input_len - input_pos);
        uint64_t num_groups =
            input_len / GROUP_LEN + (input_len % GROUP_LEN != 0);
        printf("chunk len %lu, running %lu groups\n", input_len, num_groups);
        for (int i = 0; i < 20; i++) {
            printf("%2x ", full_input[input_pos + i]);
        }
        printf("\n");
        cudaMemcpy(
            d_input,
            &full_input[input_pos],
            input_len,
            cudaMemcpyHostToDevice
        );
        printf("Copied in input\n");
        // TODO: figure out good block/thread counts
        cudaDeviceSynchronize();
        calculate_len<<<1024, 256>>>(
            d_input,
            input_len,
            d_dictionary,
            m_groups
        );
        cudaDeviceSynchronize();
        // calculate_len_cpu(
        //     &full_input[input_pos],
        //     input_len,
        //     &huffmanDictionary,
        //     m_groups
        // );
        printf("Calculated group lengths [");
        for (int i = 0; i < 20; i++) {
            printf("%lu, ", m_groups[i]);
        }
        printf("]\n");
        uint64_t compressed_len;
        calculate_offsets(m_groups, num_groups, &compressed_len, offset % 8);
        uint64_t compressed_len_bytes =
            compressed_len / 8 + (compressed_len % 8 != 0);
        printf(
            "Calculated offsets, got full len %lu (%lu bytes) and [",
            compressed_len,
            compressed_len_bytes
        );
        for (int i = 0; i < 20; i++) {
            printf("%lu ", m_groups[i]);
        }
        printf("]\n");

        // TODO: make sure we have enough memory for the compressed array.
        // normally it's fine, but depending on the huffman coding the
        // compressed array could sometimes be bigger than the input for some
        // parts This can be achieved by further splitting it up based on the
        // offsets array

        unsigned char *d_output;
        cudaMallocManaged(&d_output, compressed_len_bytes);
        cudaMemset(d_output, 0, compressed_len_bytes);
        printf("Allocated output array, synchronizing...\n");
        cudaDeviceSynchronize();
        printf("Starting apply_compression on rank %d\n", rank);
        apply_compression<<<1024, 256>>>(
            d_input,
            input_len,
            d_dictionary,
            m_groups,
            d_output,
            0
        );
        cudaDeviceSynchronize();
        apply_compression<<<1024, 256>>>(
            d_input,
            input_len,
            d_dictionary,
            m_groups,
            d_output,
            1
        );
        // apply_compression_cpu(d_input, input_len, m_groups, d_output);
        printf("Waiting on synchronize for %d\n", rank);
        cudaDeviceSynchronize();
        printf("Finished compression kernel for %d\n", rank);

        // combined_byte is used when the boundary with a previous sequence is
        // not on a byte boundary
        unsigned char combined_byte = 0;
        if (offset % 8 != 0) {
            combined_byte = output[offset / 8];
        }
        cudaMemcpy(
            &output[offset / 8],
            d_output,
            compressed_len_bytes,
            cudaMemcpyDeviceToHost
        );
        if (offset % 8 != 0) {
            output[offset / 8] |= combined_byte;
        }
        cudaFree(d_output);
        offset += compressed_len - offset % 8;
        printf("Finished chunk\n");
    }

    cudaFree(d_input);
    cudaFree(m_groups);
    cudaFree(d_dictionary);
}
