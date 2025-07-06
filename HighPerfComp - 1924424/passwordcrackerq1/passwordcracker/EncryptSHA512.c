// EncryptSHA512.c
#include <stdio.h>
#include <unistd.h>
#include <crypt.h>

int main() {
    char password[64];
    char salt[] = "$6$AS$";  // $6$ = SHA-512 with fixed salt

    printf("Enter password: ");
    fflush(stdout); // ensure the prompt is shown immediately

    if (scanf("%63s", password) != 1) {
        fprintf(stderr, "Failed to read password input.\n");
        return 1;
    }

    char *hash = crypt(password, salt);
    if (hash == NULL) {
        perror("crypt");
        return 1;
    }

    printf("Hashed password: %s\n", hash);
    return 0;
}
