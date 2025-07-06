#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// Structure to hold data for each thread
typedef struct {
    double **A;
    double **B;
    double **C;
    int r1, c1, c2;
    int row_start;
    int row_end;
} thread_data_t;

void *multiply_rows(void *arg) {
    thread_data_t *td = (thread_data_t *)arg;
    for (int i = td->row_start; i < td->row_end; i++) {
        for (int j = 0; j < td->c2; j++) {
            double sum = 0.0;
            for (int k = 0; k < td->c1; k++) {
                sum += td->A[i][k] * td->B[k][j];
            }
            td->C[i][j] = sum;
        }
    }
    return NULL;
}
//Thomas Holloway - 1924424

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input-file> <num-threads>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    char *infile = argv[1];
    int max_threads = atoi(argv[2]);
    
    FILE *fp = fopen("matrixes.txt", "r");
    if (!fp) {
        perror("Error opening input file");
        return EXIT_FAILURE;
    }

    FILE *ofp = fopen("results.txt", "w");
    if (!ofp) {
        perror("Error creating output file");
        fclose(fp);
        return EXIT_FAILURE;
    }

    int r1, c1, r2, c2;
    while (fscanf(fp, "%d,%d", &r1, &c1) == 2) {
        // Allocate A
        double **A = malloc(r1 * sizeof(double*));
        for (int i = 0; i < r1; i++) {
            A[i] = malloc(c1 * sizeof(double));
            for (int j = 0; j < c1; j++) {
                fscanf(fp, "%lf", &A[i][j]);
                // skip comma or newline
                fgetc(fp);
            }
        }
        
        // Read B dims
        if (fscanf(fp, "%d,%d", &r2, &c2) != 2) break;
        
        if (c1 != r2) {
            fprintf(stderr, "Skipping incompatible %dx%d and %dx%d\n", r1, c1, r2, c2);
            // consume B data
            for (int i = 0; i < r2; i++) {
                for (int j = 0; j < c2; j++) {
                    double tmp;
                    fscanf(fp, "%lf", &tmp);
                    fgetc(fp);
                }
            }
            // free A
            for (int i = 0; i < r1; i++) free(A[i]);
            free(A);
            continue;
        }

        // Allocate B
        double **B = malloc(r2 * sizeof(double*));
        for (int i = 0; i < r2; i++) {
            B[i] = malloc(c2 * sizeof(double));
            for (int j = 0; j < c2; j++) {
                fscanf(fp, "%lf", &B[i][j]);
                fgetc(fp);
            }
        }

        // Allocate C
        double **C = malloc(r1 * sizeof(double*));
        for (int i = 0; i < r1; i++) {
            C[i] = calloc(c2, sizeof(double));
        }

        // Determine thread count
        int nthreads = max_threads;
        if (nthreads > r1) nthreads = r1;

        pthread_t threads[nthreads];
        thread_data_t td[nthreads];

        int rows_per = r1 / nthreads;
        int extra = r1 % nthreads;
        int start = 0;

        // Launch threads
        for (int t = 0; t < nthreads; t++) {
            int end = start + rows_per + (t < extra ? 1 : 0);
            td[t].A = A;
            td[t].B = B;
            td[t].C = C;
            td[t].r1 = r1;
            td[t].c1 = c1;
            td[t].c2 = c2;
            td[t].row_start = start;
            td[t].row_end = end;
            pthread_create(&threads[t], NULL, multiply_rows, &td[t]);
            start = end;
        }

        // Join threads
        for (int t = 0; t < nthreads; t++) {
            pthread_join(threads[t], NULL);
        }

        // Write result
        fprintf(ofp, "%d,%d\n", r1, c2);
        for (int i = 0; i < r1; i++) {
            for (int j = 0; j < c2; j++) {
                fprintf(ofp, "%f%s", C[i][j], (j+1 < c2) ? "," : "\n");
            }
        }

        // Clean up
        for (int i = 0; i < r1; i++) free(A[i]);
        free(A);
        for (int i = 0; i < r2; i++) free(B[i]);
        free(B);
        for (int i = 0; i < r1; i++) free(C[i]);
        free(C);
    }

    fclose(fp);
    fclose(ofp);
    return EXIT_SUCCESS;
}
