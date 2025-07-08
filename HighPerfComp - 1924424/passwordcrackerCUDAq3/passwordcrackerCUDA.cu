#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define PASSWORD_LENGTH 4
#define ENCRYPTED_LENGTH 10
#define TOTAL_COMBINATIONS (26 * 26 * 10 * 10) // 67,600 combinations

// Device function to encrypt password using the original algorithm
__device__ void deviceCrypt(char* rawPassword, char* newPassword) {
    newPassword[0] = rawPassword[0] + 3;
    newPassword[1] = rawPassword[0] - 2;
    newPassword[2] = rawPassword[0] + 1;
    newPassword[3] = rawPassword[1] + 1;
    newPassword[4] = rawPassword[1] - 2;
    newPassword[5] = rawPassword[1] - 3;
    newPassword[6] = rawPassword[2] + 1;
    newPassword[7] = rawPassword[2] - 2;
    newPassword[8] = rawPassword[3] + 4;
    newPassword[9] = rawPassword[3] - 3;
    newPassword[10] = '\0';

    for(int i = 0; i < 10; i++) {
        if(i >= 0 && i < 6) { // checking all lower case letter limits
            if(newPassword[i] > 122) {
                newPassword[i] = (newPassword[i] - 122) + 97;
            } else if(newPassword[i] < 97) {
                newPassword[i] = (97 - newPassword[i]) + 97;
            }
        } else { // checking number section
            if(newPassword[i] > 57) {
                newPassword[i] = (newPassword[i] - 57) + 48;
            } else if(newPassword[i] < 48) {
                newPassword[i] = (48 - newPassword[i]) + 48;
            }
        }
    }
}

// Device function to generate password candidate based on thread index
__device__ void generatePasswordCandidate(int index, char* password) {
    // Extract components from the index
    int letter1_idx = (index / (26 * 10 * 10)) % 26;
    int letter2_idx = (index / (10 * 10)) % 26;
    int digit1_idx = (index / 10) % 10;
    int digit2_idx = index % 10;
    
    // Generate password: 2 letters + 2 digits
    password[0] = 'a' + letter1_idx;
    password[1] = 'a' + letter2_idx;
    password[2] = '0' + digit1_idx;
    password[3] = '0' + digit2_idx;
    password[4] = '\0';
}

// Device function to compare two strings
__device__ bool deviceStringCompare(char* str1, char* str2, int length) {
    for(int i = 0; i < length; i++) {
        if(str1[i] != str2[i]) {
            return false;
        }
    }
    return true;
}

// CUDA kernel for password cracking
__global__ void crackPasswordKernel(char* target_encrypted, char* result, int* found, int total_combinations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't exceed the total number of combinations
    if(idx >= total_combinations) return;
    
    // Each thread tries a different password combination
    char candidate[PASSWORD_LENGTH + 1];
    char encrypted_candidate[ENCRYPTED_LENGTH + 1];
    
    // Generate password candidate based on thread index
    generatePasswordCandidate(idx, candidate);
    
    // Encrypt the candidate password
    deviceCrypt(candidate, encrypted_candidate);
    
    // Compare with target encrypted password
    if(deviceStringCompare(encrypted_candidate, target_encrypted, ENCRYPTED_LENGTH)) {
        // If match found, store the result using atomic operation
        if(atomicCAS(found, 0, 1) == 0) {
            for(int i = 0; i < PASSWORD_LENGTH; i++) {
                result[i] = candidate[i];
            }
            result[PASSWORD_LENGTH] = '\0';
        }
    }
}

// Host function to encrypt password (for testing and generating target)
void hostCrypt(char* rawPassword, char* newPassword) {
    newPassword[0] = rawPassword[0] + 3;
    newPassword[1] = rawPassword[0] - 2;
    newPassword[2] = rawPassword[0] + 1;
    newPassword[3] = rawPassword[1] + 1;
    newPassword[4] = rawPassword[1] - 2;
    newPassword[5] = rawPassword[1] - 3;
    newPassword[6] = rawPassword[2] + 1;
    newPassword[7] = rawPassword[2] - 2;
    newPassword[8] = rawPassword[3] + 4;
    newPassword[9] = rawPassword[3] - 3;
    newPassword[10] = '\0';

    for(int i = 0; i < 10; i++) {
        if(i >= 0 && i < 6) { // checking all lower case letter limits
            if(newPassword[i] > 122) {
                newPassword[i] = (newPassword[i] - 122) + 97;
            } else if(newPassword[i] < 97) {
                newPassword[i] = (97 - newPassword[i]) + 97;
            }
        } else { // checking number section
            if(newPassword[i] > 57) {
                newPassword[i] = (newPassword[i] - 57) + 48;
            } else if(newPassword[i] < 48) {
                newPassword[i] = (48 - newPassword[i]) + 48;
            }
        }
    }
}

int main() {
    // Test password to crack (2 lowercase letters + 2 numbers)
    char original_password[] = "ae86";
    
    // Generate encrypted version for testing
    char target_encrypted[ENCRYPTED_LENGTH + 1];
    hostCrypt(original_password, target_encrypted);
    
    printf("Original password: %s\n", original_password);
    printf("Encrypted password: %s\n", target_encrypted);
    printf("Total combinations to check: %d\n", TOTAL_COMBINATIONS);
    
    // CUDA configuration
    int threads_per_block = 256;
    int blocks = (TOTAL_COMBINATIONS + threads_per_block - 1) / threads_per_block;
    
    printf("Using %d blocks with %d threads per block\n", blocks, threads_per_block);
    printf("Total threads: %d\n", blocks * threads_per_block);
    
    // Allocate memory on GPU
    char *d_target_encrypted, *d_result;
    int *d_found;
    
    cudaMalloc((void**)&d_target_encrypted, (ENCRYPTED_LENGTH + 1) * sizeof(char));
    cudaMalloc((void**)&d_result, (PASSWORD_LENGTH + 1) * sizeof(char));
    cudaMalloc((void**)&d_found, sizeof(int));
    
    // Copy target encrypted password to GPU
    cudaMemcpy(d_target_encrypted, target_encrypted, (ENCRYPTED_LENGTH + 1) * sizeof(char), cudaMemcpyHostToDevice);
    
    // Initialize found flag to 0
    int found_flag = 0;
    cudaMemcpy(d_found, &found_flag, sizeof(int), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch kernel
    printf("Launching kernel...\n");
    cudaEventRecord(start);
    crackPasswordKernel<<<blocks, threads_per_block>>>(d_target_encrypted, d_result, d_found, TOTAL_COMBINATIONS);
    cudaEventRecord(stop);
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy results back to CPU
    char result[PASSWORD_LENGTH + 1];
    int found_result;
    
    cudaMemcpy(result, d_result, (PASSWORD_LENGTH + 1) * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(&found_result, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    printf("\nExecution time: %.2f ms\n", milliseconds);
    
    if (found_result == 1) {
        printf("Password cracked successfully!\n");
        printf("Decrypted password: %s\n", result);
        
        // Verify the result
        char verify_encrypted[ENCRYPTED_LENGTH + 1];
        hostCrypt(result, verify_encrypted);
        printf("Verification - Re-encrypted: %s\n", verify_encrypted);
        
        if (strcmp(verify_encrypted, target_encrypted) == 0) {
            printf("Verification successful!\n");
        } else {
            printf("Verification failed!\n");
        }
    } else {
        printf("Password not found. This shouldn't happen with the complete search space.\n");
    }
    
    //Thomas Holloway - 1924424
    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_target_encrypted);
    cudaFree(d_result);
    cudaFree(d_found);
    
    // Test with a few more examples
    printf("\n--- Testing with additional passwords ---\n");
    
    char test_passwords[][5] = {"ab01", "zz99", "mp47", "qw12"};
    int num_tests = 4;
    
    for(int test = 0; test < num_tests; test++) {
        printf("\nTesting password: %s\n", test_passwords[test]);
        
        // Generate encrypted version
        char test_encrypted[ENCRYPTED_LENGTH + 1];
        hostCrypt(test_passwords[test], test_encrypted);
        printf("Encrypted: %s\n", test_encrypted);
        
        // Allocate fresh GPU memory
        cudaMalloc((void**)&d_target_encrypted, (ENCRYPTED_LENGTH + 1) * sizeof(char));
        cudaMalloc((void**)&d_result, (PASSWORD_LENGTH + 1) * sizeof(char));
        cudaMalloc((void**)&d_found, sizeof(int));
        
        // Copy data to GPU
        cudaMemcpy(d_target_encrypted, test_encrypted, (ENCRYPTED_LENGTH + 1) * sizeof(char), cudaMemcpyHostToDevice);
        found_flag = 0;
        cudaMemcpy(d_found, &found_flag, sizeof(int), cudaMemcpyHostToDevice);
        
        // Launch kernel
        crackPasswordKernel<<<blocks, threads_per_block>>>(d_target_encrypted, d_result, d_found, TOTAL_COMBINATIONS);
        cudaDeviceSynchronize();
        
        // Get results
        cudaMemcpy(result, d_result, (PASSWORD_LENGTH + 1) * sizeof(char), cudaMemcpyDeviceToHost);
        cudaMemcpy(&found_result, d_found, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (found_result == 1) {
            printf("Cracked: %s\n", result);
        } else {
            printf("Failed to crack password\n");
        }
        
        // Free memory for this test
        cudaFree(d_target_encrypted);
        cudaFree(d_result);
        cudaFree(d_found);
    }
    
    printf("\nProgram completed successfully!\n");
    return 0;
}

//Note to self:nvcc -o passwordcrackerCUDA passwordcrackerCUDA.cu ./passwordcrackerCUDA
