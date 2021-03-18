#include <iostream>

using namespace std;

int main() {
    int n, m, count;
    //Enter 2 numbers: size of matrix"
    cin >> n >> m;
    cout << n << " " << m << "\n";
    //Enter count of unreachable points
    cin >> count;
    cout << count << "\n";
    for (int i = 0; i < count; i++) {
        int first, second;
        first = rand() % n + 1;
        second = rand() % m + 1;
        cout << first << " " << second << "\n";
    }
    int first_point_begin;
    int first_point_end;
    int last_point_begin;
    int last_point_end;
    first_point_begin = rand() % n + 1;
    first_point_end = rand() % m + 1;
    last_point_begin = rand() % n + 1;
    last_point_end = rand() % m + 1;
    cout << first_point_begin << " " << first_point_end << "\n";
    cout << last_point_begin << " " << last_point_end << "\n";
    return 0;
}