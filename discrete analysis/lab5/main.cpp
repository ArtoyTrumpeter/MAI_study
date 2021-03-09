#include <iostream>
#include "SuffTree.h"

using namespace NSuff;

int main() {
    std::ios_base::sync_with_stdio(false);

    std::string input;
    std::cin >> input;

    TSuffTree tree = TSuffTree(input);
    TSuffArray array(tree.CreateSuffArray(), input);

    size_t iteration = 0;
    while(std::cin >> input) {
        ++iteration;
    
        std::vector<int> result = array.Find(input);
        if (result.empty()) {
            continue;
        }

        std::cout << iteration << ": ";
        for (size_t i = 0; i < result.size() - 1; ++i) {
            std::cout << result[i] + 1 << ", ";
        }
        std::cout << *(result.end() - 1) + 1 << std::endl;

    }

    return 0;
}