#include <cstdio>
#include <iostream>
#include <climits>

int main() {
    int m[6][6];

    // Read 2D Matrix-Array
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            std:: cin >> m[i][j];
        }
    }

    // Compute the sum of hourglasses
    long temp_sum = 0, MaxSum = LONG_MIN;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            if (j + 2 < 6 && i + 2 < 6) {
                temp_sum = m[i][j] + m[i][j + 1] + m[i][j + 2] + m[i + 1][j + 1] + m[i + 2][j] + m[i + 2][j + 1] + m[i + 2][j + 2];
                if (temp_sum >= MaxSum) {
                    MaxSum = temp_sum;
                }
            }
        }
    }
    fprintf(stderr, "Max Sum: %ld\n", MaxSum);

    return 0;
}