#include <iostream>
#include <fstream>

int main() {
    srand(time(NULL));
    std::ofstream out;
    out.open("test_file.txt");
    int amount, length;
    std::cin >> length;
    std::cin >> amount;
    for (int i = 0; i < length; i++) {
        out << char ('a' + rand() % ('z' - 'a'));
    }
    out << "\n";
    for (int i = 1; i <= amount; i++) {
        for (int j = 0; j < (10); j++) {
            out << char ('a' + rand() % ('z' - 'a'));
        }
        if (i != amount) {
            out << "\n";
        } else {
            continue;
        }
    }
    out.close();
    return 0;
}