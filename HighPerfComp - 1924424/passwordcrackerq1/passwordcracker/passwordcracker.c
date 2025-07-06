// passwordcracker.c

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <crypt.h>
#include <time.h>

static char *target_hash;        // the full hash from argv
static char salt[64];            // salt extracted (e.g. "$6$xyz123$")
static volatile int found = 0;   // flag set when password is found
static char result[6];           // to store the found password safely
static int num_threads;

// Total keyspace size: 26*26*100
#define TOTAL ((26*26*100))

void *worker(void *arg) {
    long tid = (long)arg;
    long chunk = TOTAL / num_threads;
    long start = tid * chunk;
    long end   = (tid == num_threads-1 ? TOTAL : start + chunk);

    struct crypt_data cdata;
    cdata.initialized = 0;

    for (long idx = start; idx < end && !found; idx++) {
        char guess[5];
        long rem = idx;
        int nnum = rem % 100;
        rem /= 100;
        int c2 = rem % 26;
        int c1 = rem / 26;

        guess[0] = 'A' + c1;
        guess[1] = 'A' + c2;
        guess[2] = '0' + (nnum / 10);
        guess[3] = '0' + (nnum % 10);
        guess[4] = '\0';

        char *h = crypt_r(guess, salt, &cdata);

        if (h && strcmp(h, target_hash) == 0) {
            if (!__sync_lock_test_and_set(&found, 1)) {
                strcpy(result, guess);  // safe now with result[6]
                printf("[Thread %ld] FOUND password: %s\n", tid, result);
            }
            break;
        }

        
         if (idx % 10000 == 0) {
             printf("[Thread %ld] Trying: %s\n", tid, guess);
         }
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s '<SHA512-hash>'\n", argv[0]);
        return 1;
    }
    target_hash = argv[1];

    // Extract salt = everything up through the 3rd ‘$’
    {
        int dollar = 0;
        int i = 0;
        while (target_hash[i] && dollar < 3) {
            if (target_hash[i] == '$') dollar++;
            salt[i] = target_hash[i];
            i++;
        }
        salt[i] = '\0';  // terminate after copying up to 3 '$'
    }

    num_threads = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_threads < 1) num_threads = 1;

    printf("Using %d threads, searching %d candidates each (total %d)\n",
           num_threads,
           TOTAL/num_threads + (TOTAL%num_threads?1:0),
           TOTAL);

    pthread_t threads[num_threads];
    clock_t t0 = clock();

    for (long i = 0; i < num_threads; i++) {
        if (pthread_create(&threads[i], NULL, worker, (void*)i) != 0) {
            perror("pthread_create");
            exit(1);
        }
    }
    for (int i = 0; i < num_threads; i++)
        pthread_join(threads[i], NULL);

    clock_t t1 = clock();
    double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;

    if (found) {
        printf("Cracked: %s in %.3f seconds\n", result, secs);
    } else {
        printf("Password not found in search space.\n");
    }
    return 0;
}
