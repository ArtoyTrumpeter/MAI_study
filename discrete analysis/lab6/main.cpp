#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <stdlib.h>
#include <vector>

const int base = 1000000;
const int bpow = 6;

/* [-,^,*,/] and main */
// неккоректные условия в мэйне
// перевыполнение(т 4)
// плохой power

class TBigInt {
private:
    std::vector<long long> number;

    void DeleteZero() {
        while (number.size() > 1 && number.back() == 0) {
            number.pop_back();
        }
    }

public:
    TBigInt(const std::string& input) {
        std::stringstream tempstr;
        for (auto i = (long long)input.size(); i > 0; i-= bpow) {
            if (i > bpow) {
                tempstr << input.substr(i - bpow,bpow);
                long long tempnumber;
                tempstr >> tempnumber;
                number.push_back(tempnumber);
                tempstr.clear();
            } else {
                tempstr << input.substr(0,i); 
                long long tempnumber;
                tempstr >> tempnumber;
                number.push_back(tempnumber);
                tempstr.clear();
            }
        }
        DeleteZero();
    }

    TBigInt() : number(0)
    {}

    TBigInt(int n) : number(n,0)
    {}

    friend std::ostream &operator <<(std::ostream& stream, const TBigInt& other) {
        if (other.number.size() == 0) {
            return stream;
        }
        stream << other.number[other.number.size() - 1];
        for (int i = other.number.size() - 2;i >= 0;--i)
        {
            stream << std::setfill('0') << std::setw(bpow) << other.number[i];
        }
        return stream;
    }

    TBigInt operator +(const TBigInt &other) const {
        size_t size = std::max(number.size(), other.number.size());
        TBigInt result;
        long long r = 0;
        long long k = 0;
        for (size_t i = 0; i < size; i++) {
            if (number.size() <= i) {
                k = other.number[i];
            } else if (other.number.size() <= i) {
                k = number[i];
            } else {
                k = number[i] + other.number[i];
            }
            k += r;
            result.number.push_back(k % base);
            r = k / base;   
        }
        if (r != 0) {
            result.number.push_back(r);
        }
        return result;
    } 

    TBigInt operator -(const TBigInt &other) const {
        size_t size = std::max(number.size(),other.number.size());
        TBigInt result;
        long long r = 0;
        long long k = 0;// для взятия недостатка большего числа(0 или -1)
        for (size_t i = 0; i < size;i++) {
            long long res = 0;
            if (other.number.size() <= i) {
                res = number[i] + k;
            } else {
                res = number[i] - other.number[i] + k;
            }
            k = 0;
            if (res < 0) {
                res += base;
                k = -1;
            }
            r = res % base;
            result.number.push_back(r);
        }
        result.DeleteZero();
        return result;
    }

    TBigInt operator *(const TBigInt &other) const {
        size_t size = number.size() * other.number.size();
        TBigInt result(size + 1);
        long long k = 0;
        long long r = 0;
        for (size_t i = 0; i < number.size(); i++) {
            for (size_t j = 0; j < other.number.size(); j++) {
                k = other.number[j] * this->number[i] + result.number[i+j];
                r = k / base;
                result.number[i + j + 1] = result.number[i + j + 1] + r;
                result.number[i + j] = k % base;
            }
        }
        result.DeleteZero();
        return result;
    }

    TBigInt operator /(const TBigInt &other) const {
        TBigInt cv(1);
        TBigInt result(number.size());
        for (auto i = (long long)(number.size() - 1); i >= 0; --i) {
            cv.number.insert(cv.number.begin(), number[i]);
            if (!cv.number.back()) {// уборка первого нуля
                cv.number.pop_back();
            }
            long long x = 0, l = 0, r = base;
            // бинарным поиском нахождение частного
            while (l <= r) {
                long long m = (l + r) / 2;// middle
                TBigInt cur = other * TBigInt(std::to_string(m)); // находим самое приближенное число к cv
                if (cur < cv || cur == cv) {// когда нашли
                    x = m;
                    l = m + 1;
                } else {
                    r = m - 1;
                }
            }   
            // x - частное, потом отнимается от cv (other * частное(деление уголком))
            result.number[i] = x;
            cv = cv - other * TBigInt(std::to_string(x));
        }
        result.DeleteZero();
        return result;
    }

    TBigInt Power(int power) {
        // бинарное возведение в степень
        TBigInt result("1");
        while (power) {
            if (power % 2) {
                result = result * (*this);
            }
            (*this) = (*this) * (*this);
            power /= 2;
        }
        return result;
    }

    bool operator >(const TBigInt &other) const {
        if (number.size() != other.number.size()) {
            return number.size() > other.number.size();
        }
        for (auto i = (long long)(number.size() - 1); i >= 0; i--) {
            if (number[i] != other.number[i]) {
                return number[i] > other.number[i];
            }
        }
        return false;
    } 

    bool operator <(const TBigInt &other) const {
        if (number.size() != other.number.size()) {
            return number.size() < other.number.size();
        }
        for (auto i = (long long)(number.size() - 1); i >= 0; i--) {
            if (number[i] != other.number[i]) {
                return number[i] < other.number[i];
            }
        }
        return false;
    } 

    bool operator ==(const TBigInt &other) const {
        if (number.size() != other.number.size()) {
            return false;
        }
        for (auto i = (long long)(number.size() - 1); i >= 0; i--) {
            if (number[i] != other.number[i]) {
                return false;
            }
        }
        return true;
    } 
};

int main() {
    std::string fp,sp,op;
    while(std::cin >> fp >> sp >> op) {
        if (op == "+") {
            TBigInt first(fp);
            TBigInt second(sp);
            std::cout << first + second << std::endl;
        } else if (op == "-") {
            TBigInt first(fp);
            TBigInt second(sp);
            if (first < second) {
                std::cout << "Error" << std::endl;
                continue;
            }
            std::cout << first - second << std::endl;
        } else if (op == "/") {
            TBigInt first(fp);
            TBigInt second(sp);
            if (second == TBigInt(1)) {
                std::cout << "Error" << std::endl;
                continue;
            }
            std::cout << first / second << std::endl;
        } else if (op == "*") {
            TBigInt first(fp);
            TBigInt second(sp);
            std::cout << first * second << std::endl;
        } else if (op == "^") {
            TBigInt first(fp);
            int n = atoi(sp.c_str());
            if (first == TBigInt(1) && n == 0) {
                std::cout << "Error" << std::endl;
                continue;
            }
            std::cout << first.Power(n) << std::endl;
        } else if (op == ">") {
            TBigInt first(fp);
            TBigInt second(sp);
            std::cout << ((first > second) ? "true" : "false") << std::endl;
        } else if (op == "=") {
            TBigInt first(fp);
            TBigInt second(sp);
            std::cout << ((first == second) ? "true" : "false") << std::endl;
        } else if (op == "<") {
            TBigInt first(fp);
            TBigInt second(sp);
            std::cout << ((first < second) ? "true" : "false") << std::endl;
        }
    }
    return 0;
}