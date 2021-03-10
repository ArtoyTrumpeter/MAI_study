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
    for (int i = 0; i < amount; i++) {
        for (int i = 0; i < (10); i++) {
            out << char ('a' + rand() % ('z' - 'a'));
        }
        if (i != --amount) {
            out << "\n";
        }
    }
    out.close();
    return 0;
}