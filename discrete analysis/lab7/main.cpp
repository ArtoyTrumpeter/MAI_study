#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <stdlib.h>
#include <vector>

// ошибки на чекере из-за перевыполнения
long long Minimum(long long first, long long second, long long third) {
    if(first <= second && first <= third) {
        return first;
    } else if(first >= second && second <= third) {
        return second;
    } else if(first >= third && third <= second) {
        return third;
    }
    return 0;
}


int main() {
    // initialization
    long long **A;
    long long **B;
    int n,m;
    std::cin >> n >> m;

    int i = 0, j = 0, k = 0;
    A = new long long*[n];
    for(i = 0; i < n; i++) {
        A[i] = new long long[m];
    }
    for(i = 0; i < n; i++) {
        for(j = 0; j < m;j++) {
            std::cin >> A[i][j]; 
        }
    }
    B = new long long*[n];
    for(i = 0; i < n; i++) {
        B[i] = new long long[m];
    }
    for(i = 0; i < n; i++) {
        for(j = 0; j < m;j++) {
            B[i][j] = 0; 
        }
    }
    // initialization
    // algorithm
    for(j = 0; j < m; j++) {
        B[n - 1][j] = A[n - 1][j];
    }
    for(i = n - 2; i >= 0; i--) {
        B[i][0] = (Minimum(B[i + 1][0], B[i + 1][0], B[i + 1][1]) + A[i][0]);
        for(j = 1; j < m - 1; j++) {
            B[i][j] = (Minimum(B[i + 1][j - 1], B[i + 1][j], B[i + 1][j + 1]) + A[i][j]);
        }
        B[i][m - 1] = (Minimum(B[i + 1][m - 2], B[i + 1][m - 1], B[i + 1][m - 1]) + A[i][m - 1]);
    }
    // algorithm
    // output
    for(j = 1; j < m; j++) {
        if(B[0][k] > B[0][j]) {
            k = j;
        }
    }
    std::cout << B[0][k] << "\n " << "(1," <<  k + 1 << ") ";
    for(i = 1; i < n; i++) {
        if((k > 0) && (B[i][k - 1] < B[i][k])) {
            if((k + 1 < m) && (B[i][k - 1] > B[i][k + 1])) {
                k++;
            } else {
                k--;
            }
        } else {
            if((k + 1 < m) && (B[i][k + 1] < B[i][k])) {
                k++;
            }
        }
        std::cout << "("<< i + 1 << "," <<  k + 1 << ") ";
    }
    std::cout << "\n";
    // output
    // clear
    for(int i = 0; i < n; i++) {
        delete [] A[i];
        delete [] B[i];
    }
    delete [] A;
    delete [] B;
    // clear
    return 0;
}