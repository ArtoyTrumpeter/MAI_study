#include "source/CfreeG.hpp"

void PrintRules() {
    std::cout << "Context-free grammar" << std::endl;
    std::cout << "Infinity:&" << std::endl;
    std::cout << "long long int - type" << std::endl;
    std::cout << "S->{A" << std::endl;
    std::cout << "A->[a,b]B || (a,b)B ||[a,b)B || (a,b]B || }" << std::endl;
    std::cout << "B->,A || } " << std::endl;
}


void PrintMain() {
    std::cout << "Enter go:work" << std::endl;
    std::cout << "Enter end:exit" << std::endl;
    std::cout << "Enter help:help" << std::endl;
}


//  ConvertInString
int main() {
    std::string out;
    PrintRules();
    while(std::cin >> out) {
        if(out == "go") {
            try {
                std::string line;
                std::cin >> line;
                ContextFree* current = new ContextFree(line);
                current->CheckNormalize();
                delete current;
            } catch (std::logic_error& err) {
                std::cout << err.what() << std::endl;
            }
        } else if(out == "end") {
            break;
        } else if(out == "help") {
            PrintMain();
        }
    }    
    return 0;
}