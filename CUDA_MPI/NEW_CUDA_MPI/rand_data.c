#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUMTHREADS 12

/**
 * @brief Parses an argument containing a file size
 *
 * Examples:
 * ```
 * parse_size("100") == 100;
 * parse_size("2KB") == 2*1024;
 * parse_size("1.5MB") == (size_t)(1.5*1024.0*1024.0);
 * parse_size("20GB") == 20*1024*1024*1024;
 * parse_size("asdf") == 0;
 * ```
 *
 * @param size_arg The null-terminated input string
 * @return size_t The size in bytes, or `0` if invalid.
 */
size_t parse_size(char *size_arg) {
    size_t len = strnlen(size_arg, 1000);
    if (len == 1000) {
        // idk max size number length I guess
        return 0;
    }

    if (size_arg[len - 1] == 'B' || size_arg[len - 1] == 'b') {
        switch (size_arg[len - 2]) {
        case 'K':
        case 'k': {
            char *end_ptr;
            double size = strtod(size_arg, &end_ptr);
            if (end_ptr != &size_arg[len - 2] || size <= 0.0) {
                return 0;
            }
            return (size_t)(size * 1024.0);
            break;
        }
        case 'M':
        case 'm': {
            char *end_ptr;
            double size = strtod(size_arg, &end_ptr);
            if (end_ptr != &size_arg[len - 2] || size <= 0.0) {
                return 0;
            }
            return (size_t)(size * 1024.0 * 1024.0);
            break;
        }
        case 'G':
        case 'g': {
            char *end_ptr;
            double size = strtod(size_arg, &end_ptr);
            if (end_ptr != &size_arg[len - 2] || size <= 0.0) {
                return 0;
            }
            return (size_t)(size * 1024.0 * 1024.0 * 1024.0);
            break;
        }
        default: return 0; // invalid
        }
    }
    // otherwise just in bytes
    return strtoull(size_arg, NULL, 10);
}

struct fill_rand_args {
    /// [out] the buffer to fill with random data (should have space to be
    /// short-aligned, so may need an extra byte allocated).
    char *buf;
    /// The length of `buf`.
    size_t len;
    /// The seed to be used on this thread.
    unsigned int seed;
};

/**
 * @brief Fills a buffer with random data, intended for use with `pthreads`
 *
 * @param args Should be a pointer to a `fill_rand_args`
 * @return void* Returns `NULL`
 */
void *fill_rand(void *args) {
    for (size_t i = 0; i < ((struct fill_rand_args *)args)->len;
         i += sizeof(short)) {
        *((short *)&((struct fill_rand_args *)args)->buf[i]) =
            (short)rand_r(&((struct fill_rand_args *)args)->seed);
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(
            stderr,
            "Usage: %s <file> <size>\n"
            "Size can also be \"<number>KB\", \"<number>MB\", or "
            "\"<number>GB\"\n",
            argv[0]
        );
        return EXIT_FAILURE;
    }
    srand(time(NULL));

    size_t size = parse_size(argv[2]);
    if (size == 0) {
        fprintf(stderr, "ERROR: invalid file size.\n");
        return EXIT_FAILURE;
    }

    FILE *file = fopen(argv[1], "w");
    char *buf = malloc(
        size + sizeof(int)
    ); // bit extra so our rngs can just write to it

    pthread_t threads[NUMTHREADS - 1];
    struct fill_rand_args args[NUMTHREADS];
    for (size_t t = 0; t < NUMTHREADS - 1; t++) {
        args[t].buf = &buf[(size / NUMTHREADS) * t];
        args[t].len = size / NUMTHREADS;
        args[t].seed = (unsigned int)rand();

        pthread_create(&threads[t], NULL, &fill_rand, &args[t]);
    }
    args[NUMTHREADS - 1].buf = &buf[(size / NUMTHREADS) * (NUMTHREADS - 1)];
    args[NUMTHREADS - 1].len = size - NUMTHREADS * (size / NUMTHREADS);
    args[NUMTHREADS - 1].seed = (unsigned int)rand();
    fill_rand(&args[NUMTHREADS - 1]);
    for (size_t t = 0; t < NUMTHREADS - 1; t++) {
        pthread_join(threads[t], NULL);
    }
    fwrite(buf, 1, size, file);
    fclose(file);
}
