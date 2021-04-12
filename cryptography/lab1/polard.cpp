#include <iostream>
#include <string>
#include <gmpxx.h>
#include <vector>
#include <ctime>


class po_Polard{
public:
    po_Polard(const mpz_class& num);
    po_Polard(const std::string& num);
    mpz_class get_factor();
    void factor_list(std::vector<mpz_class>& factors);
private:
    void add_factors(std::vector<mpz_class>& factors, mpz_class num);
    mpz_class factor_of_num(mpz_class& num);
    void Polard_GCD(mpz_class& ans, mpz_class& x, mpz_class& y);
    void Polard_MOD(mpz_class& ans, mpz_class& x, mpz_class& y);
    void Polard_ABSOLUTE(mpz_class& ans, mpz_class& x, mpz_class& y);
    mpz_class number;
};


void po_Polard::factor_list(std::vector<mpz_class>& factors){
    factors.clear();
    add_factors(factors, number);
}

void po_Polard::add_factors(std::vector<mpz_class>& factors, mpz_class num){
    while(num > 1){
        mpz_class fact = factor_of_num(num);
        //std::cout << fact << ' ' << num << std::endl;
        if(fact == num){
            factors.push_back(fact);
            return;
        }
        add_factors(factors, fact);
        num /= fact;
    }
}

mpz_class po_Polard::factor_of_num(mpz_class& num){
    mpz_class x, y, ans, absolute;
    unsigned long long i = 0, stage = 2;
    x = (rand() % (number - 1)) + 1;
    y = 1;
    Polard_ABSOLUTE(absolute, x, y);
    Polard_GCD(ans, num, absolute);
    while(ans == 1){
        if(i == stage){
            y = x;
            stage <<= 1;
        }
        absolute = x * x + 1;
        Polard_MOD(x, absolute, num);
        ++i;
        Polard_ABSOLUTE(absolute, x, y);
        Polard_GCD(ans, num, absolute);
    }
    return ans;
}

mpz_class po_Polard::get_factor(){
    return factor_of_num(number);
}


po_Polard::po_Polard(const mpz_class& num){
    srand(time(0));
    number = num;
}

po_Polard::po_Polard(const std::string& str){
    srand(time(0));
    number = str;
}

void po_Polard::Polard_ABSOLUTE(mpz_class& ans, mpz_class& x, mpz_class& y){
    x -= y;
    mpz_abs(ans.get_mpz_t(), x.get_mpz_t());
    x += y;
}


void po_Polard::Polard_GCD(mpz_class& ans, mpz_class& x, mpz_class& y){
    mpz_gcd(ans.get_mpz_t(), x.get_mpz_t(), y.get_mpz_t());
}

void po_Polard::Polard_MOD(mpz_class& ans, mpz_class& x, mpz_class& y){
    mpz_mod(ans.get_mpz_t(), x.get_mpz_t(), y.get_mpz_t());
}

using namespace std;

int main(){
    std::string number = "352358118079150493187099355141629527101749106167997255509619020528333722352217";
    po_Polard polard(number);
    std::cout << "Factor: " << polard.get_factor() << endl;
    /*
    std::vector<mpz_class> factors;
    polard.factor_list(factors);
    std::cout << "Factors of " << number << ':' << std::endl;
    for(unsigned i = 0; i < factors.size(); ++i){
        std::cout << factors[i] << std::endl;
    }
    */
    return 0;
}