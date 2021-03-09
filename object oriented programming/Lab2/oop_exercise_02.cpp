    
/* create class Modulo doing simple mathematic operations with int numbers
   Тояков Артем Олегович М8о-207Б*/
#include <iostream>
#include <stdlib.h>

using namespace std;

class Modulo
{
    int number, N;

    public:

    Modulo (int a, int b) {
        number = a;
        N = b;
    }
    
    int operator + (const Modulo B) {
        int result = ((this->number % B.N) + (B.number % B.N)) % B.N;
        return result;
    }

    int operator - (const Modulo B) {
        int result = ((this->number % B.N) - (B.number % B.N)) % B.N;
        return result;
    }

    int operator * (const Modulo B) {
        int result = ((this->number % B.N) * (B.number % B.N)) % B.N;
        return result;
    }

    int operator / (const Modulo B) {
        int result = ((this->number % B.N) / (B.number % B.N)) % B.N;
        return result;
    }

    Modulo operator >(const Modulo B) {
        if ((this->number % this->N) > (B.number % B.N)) {
            cout << "First number greater than second" << "\n";
        } else if ((this->number % this->N) < (B.number % B.N)) {
            cout << "First number less than second" << "\n";
        } else {
            cout << "Numbers are equal" << "\n";
        }
        return *this;
    }
};


Modulo operator "" _modulo_numbers(const char* str, size_t size) {
    size_t i = 0;
    int rez, mod;
    string a;
    string b;
    while (str[i] != ' ') {
        a = a + str[i];
        i++;
    }
    rez = stoi(a, 0, 10);
    i++;
    while (str[i] != '\0') {
        b = b + str[i];
        i++;
    }
    mod = stoi(b, 0, 10);
    return Modulo(rez, mod);
}

int main() {
    Modulo A = "56446 956"_modulo_numbers;
    Modulo B = "1684 956"_modulo_numbers;
    cout << A+B << "\n";
    cout << A-B << "\n";
    cout << A*B << "\n";
    cout << A/B << "\n";
    A>B;
    return 0;
}