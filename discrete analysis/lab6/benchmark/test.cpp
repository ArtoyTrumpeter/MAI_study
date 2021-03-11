#include <iostream>


int main() {
    srand(time(NULL));
    unsigned long long  n;
    std::string operation;
    std::cin >> operation;
    std::cin >> n;
    for(auto i = 0; i < n; i++) {
        unsigned long long size;
        size = rand() % 10000 + 1;
        for(auto i = 0; i < size;i++) {
            std::cout << rand() % 9; 
        }
        std:: cout << "\n";
        size = rand() % 10000 + 1;
        for(auto i = 0; i < size;i++) {
            std::cout << rand() % 9; 
        }
        std:: cout << "\n";
        std::cout << operation << "\n";
    }
    return 0;
}