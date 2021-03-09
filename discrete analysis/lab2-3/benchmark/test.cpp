#include <iostream>

int main() {
    int size;
    std::cin >> size;
    int sizestr;
    std::cin >> sizestr;
    std::cout << size << "\n";
    std::cout << sizestr << "\n";
    int a = 5355345; 
    srand(time(NULL));
    for(int i = 0; i < size; i++) {
        for(int i = 0; i < sizestr; i++) {
            std::cout << char ('a' + rand() % ('z' - 'a'));
        }
        std::cout << " " << a << "\n";
    }
    return 0;
}