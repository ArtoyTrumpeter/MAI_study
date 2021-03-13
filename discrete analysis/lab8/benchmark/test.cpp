#include <iostream>

int main() {
    int n, m;
    srand(time(NULL));
    std::cin >> n;
    std::cout << n << "\n"; 
    for(int i = 0; i < n; i++) {
        int left = -(rand() % 20000) + 10000;
        int right = -(rand() % 20000) + 10000;
        std::cout << left << " " << right;
        std::cout << "\n";
    }
    m = rand() % 1000;
    std::cout << m << "\n"; 
    return 0;
}