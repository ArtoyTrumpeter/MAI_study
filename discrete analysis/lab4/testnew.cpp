#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <ctime>

using namespace std;

vector <long long> z_function(string s) {
    int n = (int) s.length();
    vector <long long> z(n, 0);
    long long right = 0, left = 0;
    vector <long long> reverse (n, 0);
    for (int i = 0; i < n - 1; i++) {
        reverse[i] = s[n - 1 - i];
    }
    for (long long j = 1; j < n; ++j) {
        if (j <= right)
            z[j] = min(right - j + 1, z[j - left]);
        while ((j + z[j] < n) && (reverse[z[j]] == reverse[j + z[j]])) {
            z[j]++;
        }
        if ((j + z[j] - 1) > right) {
            left = j;
            right = j + z[j] - 1;
        }
    }
    /*for (int i = 0; i < n - 1; i++) {
        reverse[i] = z[i];
    }
    for (int i = 0; i < n - 1; i++) {
        z[i] = reverse[n - 1 - i];
    }*/
    return z;
}

vector <long long> GoodSuff(vector <long long> &suff, string s) {
    int n = (int) s.length();
    vector <long long> z(n, 0);
    z = z_function(s);
    for (long long j = n - 1; j > 0; j--) {
        suff[n - z[j]] = j;
    }
    for (long long j = 1, r = 0; j < n; j++) {
        if (j + z[j] == n) {
            for (; r <= j; r++) {
                if (suff[r] == n) {
                    suff[r] = j;
                }
            }
        }
    }
    return suff;
}

int main() {
    string s;
    cin >> s;
    int n = (long) s.length();
    vector <long long> answer = z_function(s);
    vector <long long> suff(n + 1, n);
    vector <long long> answer2 = GoodSuff(suff, s);
    for (int i = 0; i < answer.size(); i++) {
        cout << answer.at(i) << ' ';
    }
    cout << endl;
    for (int i = 0; i < answer2.size(); i++) {
        cout << answer2.at(i) << ' ';
    }
    return 0;
}