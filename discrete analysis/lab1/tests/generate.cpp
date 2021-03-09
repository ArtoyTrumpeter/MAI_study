#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string.h>
#include <climits>

using namespace std;

int main() {
    srand(0);
    ofstream file("file2.txt", ios::app);
    const char alphanum[] = "abcdefghigklmnopqrstuvwxyz";
    if (!file) {
        exit (1);
    }
    for(int i = 0; i < 80000; i++) {
        int a = rand() % 30000;
        char b[60];
        for (int i = 0; i < 60; i++) {
            b[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
        }
        file << a  << " " << b << "\n";
    }
    file.close();
}