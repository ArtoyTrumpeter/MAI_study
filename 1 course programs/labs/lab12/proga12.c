#include <stdio.h>


int get_capacity(int number) {
    int answer = 1;
    while ((answer * 10) <= number) {
        answer *= 10;
    }
    return answer;
}


int func(int number) {
    int sign_value = 1;
    if (number < 0) {
        number *= -1;
        sign_value = -1;
    }
    int second_digit, digit_before_last, capacity = get_capacity(number);
    if ((number / 10) != 0) {
        second_digit = (number / (capacity / 10)) % 10;
        digit_before_last = (number / 10) % 10;
        number -= (capacity / 10) * second_digit;
        number += (capacity / 10) * digit_before_last;
        number -= 10 * digit_before_last;
        number += 10 * second_digit;
    }
    return (number * sign_value);
}


int main(void) {
    int number;
    while (scanf("%d", &number) == 1) {
        printf("%d\n", func(number));
    }
    return 0;
}