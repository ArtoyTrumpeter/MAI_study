#include <iostream>

int main() {
    int n, m;
    srand(time(NULL));
    std::cin >> n >> m;
    std::cout << n << " " << m << "\n"; 
    for (int i = 0; i < m; i++) {
        int left = rand() % n + 1;
        int right = rand() % n + 1;
        std::cout << left << " " << right;
        std::cout << "\n";
    }
    return 0;
}