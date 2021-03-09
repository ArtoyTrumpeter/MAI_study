/* create class Modulo doing simple mathematical operations with int numbers
  Тояков Артем Олегович М8о-207Б*/

#include <iostream>

using namespace std;

class Modulo
{
    private:
        int result;
        int number, module;
    public:
        Modulo (int number, int module) {
            this->number = number;
            this->module = module;
        }
        
        void sum(Modulo & B) {
            result = (this->number % this->module + B.number % B.module) % this->module;
        }
        int get_sum() {
            return result;
        }
        
        void diff(Modulo & B) {
            result = (this->number % this->module - B.number % B.module) % this->module;
        }
        int get_diff() {
            return result;
        }
        
        void mult(Modulo & B) {
            result = (this->number % this->module * B.number % B.module) % this->module;
        }
        int get_mult() {
            return result;
        }
        
        void division(Modulo & B) {
            result = (this->number % this->module / B.number % B.module) % this->module;
        }
        int get_division() {
            return result;
        }
        
        void comparison(Modulo & B) {
            if ((this->number % this->module) > (B.number % B.module)) {
                cout << "First number greater than second" << "\n";
            } else if ((this->number % this->module) < (B.number % B.module)) {
                cout << "First number less than second" << "\n";           
            } else {
                cout << "Numbers are equal" << "\n";
            }
        }

        int Get_number() {
            return this->number;
        }
};

int main() {
    int number1, number2, module_N;
    cout << "Enter two numbers and module" << "\n";
    cin >> number1 >> number2 >> module_N;
    while (module_N == 0) {
        cout << "Enter correct module" << "\n";
        cin >> module_N;
    }
    Modulo A(number1, module_N);
    Modulo B(number2, module_N);
    A.sum(B);
    cout << "number1 = " << number1 << " number2 = " << number2 <<
        " module_N = " << module_N << "\n" << "sum = " << A.get_sum() << "\n";
    A.diff(B);
    cout << "diff = " << A.get_diff() << "\n";
    A.mult(B);
    cout << "mult = " << A.get_mult() << "\n";
    if ((number2 % module_N) == 0) {
        cout << "Can't do division" << "\n";
    } else {
        A.division(B);
        cout << "division = " << A.get_division() << "\n";
    }
    A.comparison(B);
    return 0;
}
