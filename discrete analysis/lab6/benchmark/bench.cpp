#include "BigInt.hpp"

int main() {
    double top = 0;
    double start, end;
    /* double startall, endall;
    startall = clock()
    */
    std::string fp,sp,op;
    while(std::cin >> fp >> sp >> op) {
        if (op == "+") {
            TBigInt first(fp);
            TBigInt second(sp);
            start = clock();
            first + second;
            end = clock();
            top += (end - start);
        } else if(op == "-") {
            TBigInt first(fp);
            TBigInt second(sp);
            if(first < second) {
                std::cout << "Error" << std::endl;
                continue;
            }
            start = clock();
            first - second;
            end = clock();
            top += (end - start);
        } else if(op == "/") {
            TBigInt first(fp);
            TBigInt second(sp);
            if(second == TBigInt(1)) {
                std::cout << "Error" << std::endl;
                continue;
            }
            start = clock();
            first / second;
            end = clock();
            top += (end - start);
        } else if(op == "*") {
            TBigInt first(fp);
            TBigInt second(sp);
            start = clock();
            first * second;
            end = clock();
            top += (end - start);
        } else if(op == "^") {
            TBigInt first(fp);
            int n = atoi(sp.c_str());
            if(first == TBigInt(1) && n == 0) {
                std::cout << "Error" << std::endl;
                continue;
            }
            start = clock();
            first.Power(n);
            end = clock();
            top += (end - start);
        } else if(op == ">") {
            TBigInt first(fp);
            TBigInt second(sp);
            start = clock();
            std::cout << ((first > second) ? "true" : "false") << std::endl;
            end = clock();
            top += (end - start);
        } else if(op == "=") {
            TBigInt first(fp);
            TBigInt second(sp);
            start = clock();
            std::cout << ((first == second) ? "true" : "false") << std::endl;
            end = clock();
            top += (end - start);
        } else if(op == "<") {
            TBigInt first(fp);
            TBigInt second(sp);
            start = clock();
            std::cout << ((first < second) ? "true" : "false") << std::endl;
            end = clock();
            top += (end - start);
        }
    }
    // endall = clock()
    // std::cout << "Time of working " << (endall - startall) / CLOCKS_PER_SEC << "sec" << std::endl;
    std::cout << "Time of working " << top / CLOCKS_PER_SEC << "sec" << std::endl;
    return 0;
}