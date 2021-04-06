#ifndef CFREEG_HPP
#define CFREEG_HPP
#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include "line.hpp"
#include <stdlib.h>

// infinity &;
/*
Context-free grammar
    S->{A
    A->[a,b]B|(a,b)B|[a,b)B|(a,b]B|};
    B->,A|}
*/

class ContextFree {
public:
    ContextFree(const std::string& str) {
        //PrintRules();
        if(LLParsing(str) == false) {
            throw std::logic_error("Pattern is wrong");
        }
        line = std::move(str);
        normalize = false;
    }


    void PrintRules() {
        std::cout << "Context-free grammar" << std::endl;
        std::cout << "Infinity:&" << std::endl;
        std::cout << "long long int -` type" << std::endl;
        std::cout << "S->{A" << std::endl;
        std::cout << "A->[a,b]B || (a,b)B ||[a,b)B || (a,b]B || }" << std::endl;
        std::cout << "B->,A || } " << std::endl;
    }


    // if(normalize == true) -> nothing, else normalize line and print this line!
    void CheckNormalize() {
        if(normalize == false) {
            Normalize(line);
        }
        std::cout << "Normalize: " << std::endl;
        std::cout << line << std::endl;
    }


    ~ContextFree() {
        line = "";
        pairs.clear();
        startPositions.clear();
        normalize = false;
    }
private:
    std::string line;
    std::vector<std::shared_ptr<Line<long long>>> pairs;
    std::vector<std::pair<size_t,size_t>> startPositions;
    bool normalize;
    enum States {
        S,
        A,
        B,
        End,
        Error,
    };
    enum ConstStates {
        FD,
        SD,
        EndB,
        ErrorB
    };
    
    // 2
    // add 
    void AddInVector(std::string& str) {
        std::shared_ptr<Line<long long>> current(new Line<long long>(str));
        auto it = pairs.begin();
        bool merge = false;
        while((merge == false) && (it != pairs.end())) {
            merge = ((*it)->Merge(current));
            it++;
        }
        if(merge == false) {
            pairs.push_back(current);
        }
    }
    

    // sort vector
    void SortVector() {
        std::sort(pairs.begin(), pairs.end(), [](std::shared_ptr<Line<long long>>& first,
        std::shared_ptr<Line<long long>>& second) {
            return (first->a < second->a);
        });
    }


    void PrintVector() {
        std::for_each(pairs.begin(),pairs.end(),[](std::shared_ptr<Line<long long>> o)
            {
                return o->Print();
            });
    }


    void NormalizeVector() {
        if(pairs.begin() == pairs.end()) {
            return;
        }
        auto it = pairs.begin();
        auto next = pairs.begin() + 1;
        while(next != pairs.end()) {
            if((*it)->b >= (*next)->a) {
                if((*next)->b > (*it)->b) { // [1,7],[2,8] -> [1,8]
                    (*it)->b = (*next)->b;
                    (*it)->right = (*next)->right;
                } else if((*next)->b == (*it)->b) { // [1,7],[2,7] -> [1,7]
                    if((*it)->b == true || (*next)->b == true) {
                        (*it)->right = true;
                    }
                } else if((*it)->b == (*next)->a) { // [1,7],(7,13) -> [1,13)
                    if((*next)->a == true || (*it)->b == true) {
                        (*it)->b = (*next)->b;
                        (*it)->right = (*next)->right;
                    }
                }
                pairs.erase(next);
            } else { // [1,7], (8,10) -> const;
                it++;
                next++;
            }
        }
    }


    // convert
    void ConvertInString() {
        std::string newLine = "{";
        int count = 0;
        for(auto it = pairs.begin(); it != pairs.end(); it++) {
            std::string interval = "";
            if((*it)->left == true) {
                interval = interval + "[";
            } else {
                interval = interval + "(";
            }
            if((*it)->a == std::numeric_limits<long long>::min()) {
                interval = interval + "-&";
            } else {
                interval = interval + std::to_string((*it)->a);
            }
            interval = interval + ",";
            if((*it)->b == std::numeric_limits<long long>::max()) {
                interval = interval + "+&";
            } else {
                interval = interval + std::to_string((*it)->b);
            }
            if((*it)->right == true) {
                interval = interval + "]";
            } else {
                interval = interval + ")";
            }
            count++;
            if(count != 1) {
                newLine = newLine + ",";
                newLine = newLine + interval;
            } else {
                newLine = newLine + interval; 
            }
        }
        newLine += "}";
        line = std::move(newLine);
    }


    // all
    void Normalize(const std::string& str) {
        for(auto it = startPositions.begin(); it != startPositions.end(); it++) {
            std::string interval = str.substr(it->first, (it->second - it->first) + 1);
            AddInVector(interval);
        }
        SortVector();
        //PrintVector();
        NormalizeVector();
        //PrintVector();
        ConvertInString();
        normalize = true;
    }

    // 1
    // check and left and right digit in brackets!
    bool CheckInfinity(const bool& lInf,const bool& rInf,const bool& lSign,const bool& rSign) {
        if((lInf == true && lSign == true) || (rInf == true && rSign == false)) {
            return false;
        } 
        return true;
    }


    bool CheckDigit(const bool& lSign,const bool& rSign,const int& lDigit,const int& rDigit) {
        if(lDigit == 0 && rDigit == 0) {
            return false;
        }
        if(lSign == false && rSign == false) {
            if(rDigit >= lDigit) {
                return false;
            }
        } else if(lSign == true && rSign == false) {
            return false;
        } else if(lSign == true && rSign == true) {
            if(lDigit >= rDigit) {
                return false;
            }
        }
        return true;
    }


    bool CheckInfinityDigit(const bool& lSign, const bool& lInf) {
        if(lInf == true && lSign == true) {
            return false;
        } 
        return true;
    }


    bool CheckDigitInfinity(const bool& rSign, const bool& rInf) {
        if(rInf == true && rSign == false) {
            return false;
        } 
        return true;
    }


    // Get digit or infinity
    void GetDigit(const std::string& str, size_t& i, bool& sign, int& digit, bool& statusDigit) {
        if((str[i] == '-') || (str[i] == '+')) {
            if(str[i] == '-') {
                sign = false;
            }
            i++;
        }
        while((i < str.length()) && (std::isdigit(str[i]))) {
            statusDigit = true;
            char current = str[i];
            digit = digit * 10 + (current - '0');
            i++;
        } 
    }


    bool GetLowInfinity(const std::string& str, size_t& i, bool& sign, bool& inf) {
        if((i + 2 < str.length()) && ((str[i] == '-') && (str[i + 1] == '&'))) {
            inf = true;
            sign = false;
            i = i + 2;
            return true;
        }
        return false;
    }


    bool GetHighInfinity(const std::string& str, size_t& i, bool& sign, bool& inf) {
        if((i + 2 < str.length()) && ((str[i] == '+') && (str[i + 1] == '&'))) {
            inf = true;
            sign = true;
            i = i + 2;
            return true;
        }
        return false;
    }


    // Outburst [a,b] || (a,b) || [a,b) || (a,b]
    bool OutBurst(const std::string& str, size_t& i) {
        int lBracket = i;
        if((str[i] != '[') && (str[i] != '(')) {
            return false;
        }
        i++;
        ConstStates state = FD;
        bool lInf = false, rInf = false;
        bool lSign = true, rSign = true;
        bool statusLDigit = false, statusRDigit = false;
        int lDigit = 0, rDigit = 0;
        while(state != ErrorB && state != EndB) {
            switch(state) {
                case FD:
                    if(GetLowInfinity(str, i, lSign, lInf) == false) {
                       GetDigit(str, i, lSign, lDigit, statusLDigit);
                    }
                    if((str[i] == ',') && (lInf == true || statusLDigit == true)) {
                        i++;
                        state = SD;
                    } else {
                        state = ErrorB;
                    }
                    break;
                case SD:
                    if(GetHighInfinity(str, i, rSign, rInf) == false) {
                        GetDigit(str,i,rSign,rDigit,statusRDigit);
                    }
                    if((str[i] == ')' || str[i] == ']') && (rInf == true || statusRDigit == true)) {
                        startPositions.push_back(std::make_pair(lBracket,i));
                        state = EndB;
                    } else {
                        state = ErrorB;
                    }
                    break;
                case EndB:
                    break;
                case ErrorB:
                    break;
            }
        }
        if(state == ErrorB) {
            return false;
        }
        // only (-&,+&) true;
        if((str[lBracket] != '(' && lInf == true) || (str[i] != ')' && rInf == true)) {
            return false;
        }
        bool st;
        if(lInf == true && rInf == true) {
            st = CheckInfinity(lInf, rInf, lSign, rSign);
        } else if(statusLDigit == true && statusRDigit == true) {
            st = CheckDigit(lSign, rSign, lDigit, rDigit);
        } else if(lInf == true && statusRDigit == true) {
            st = CheckInfinityDigit(lSign, lInf);
        } else if(statusLDigit == true && rInf == true) {
            st = CheckDigitInfinity(rSign, rInf);
        }
        return st;
    }


    // LL-parsing;
    bool LLParsing(const std::string& str) {
        size_t length = str.size();
        size_t i = 0;
        States state = S;
        bool check;
        while(i < length) {
            char current = str[i];
            switch(state) {
                case S:
                    if(current == '{') {
                        i++;
                        state = A;
                    } else {
                        i = length - 1;
                        state = Error;
                    }
                    break;
                case A:
                    if(current == '}') {
                        state = End;
                    }
                    else if(OutBurst(str, i) == true && (i < length - 1)) {
                        i++;
                        state = B;
                    } else {
                        i = length - 1;
                        state = Error;
                    }
                    break;
                case B:
                    if(current == ',') {
                        i++;
                        state = A;
                    } else if(current == '}') {
                        state = End;
                    } else {
                        i = length - 1;
                        state = Error;
                    }
                    break;
                case End:
                    if(i == length - 1) {
                        check = true;
                        i = length;
                    } else {
                        check = false;
                        i = length;
                    }
                    break;
                case Error:
                    check = false;
                    i = length;
                    break;
            }
        }
        return check;
    }
};


#endif