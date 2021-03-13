#include <iostream>

int main() {
    int n,m;
    srand(time(NULL));
    std::cin >> n >> m;
    std::cout << n  << " " << m << "\n"; 
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            long long random = rand()%100000000;
            std::cout << random << " ";
        }
        std:: cout << "\n";
    }
    return 0;
}