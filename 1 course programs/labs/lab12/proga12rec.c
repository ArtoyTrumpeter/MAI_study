#include <stdio.h>


int change(int number) {
    int sign_value = 1;
    if (number < 0) {
        number *= -1;
        sign_value = -1;
    }
    if (number < 10) {
        return (number * sign_value);
    } else {
        int sec_digit, digit_before_last, tmp = number;
        digit_before_last = (number / 10) % 10;
        while (tmp > 99) {
            tmp /= 10;
        }
        if (tmp <= 99) {
            sec_digit = tmp % 10;
        }  
    }
}