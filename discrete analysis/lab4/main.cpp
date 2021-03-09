#include <iostream>
#include <string>
#include <vector>
#include <map>

using namespace std;

struct TWords {
    long long wrd;
    long long str;
    long long val;
};

TWords AddWord(long long wrd, long long str, long long val) {
        TWords tmp;
        tmp.wrd = wrd;
        tmp.str = str;
        tmp.val = val;
        return tmp;
}

void ParseText(vector<TWords> &text) {
    string s;
    long long str_num = 1;
    long long wrd = 1;
    long long symbol;
    bool num_read = false;
    int temp = 0;
    symbol = getchar();
    while (true) {
        if ((symbol == ' ') || (symbol == '\n')) {
            if (num_read) {
                text.push_back(AddWord(wrd, str_num, temp));
                num_read = false;
                temp = 0;
                wrd++;
            }
            if (symbol == '\n') {
                str_num++;
                wrd = 1;
            }
        } else {
            if (!num_read) {
                num_read = true;
            }
            temp = temp * 10 + (symbol - '0');
        }
        symbol = getchar();
        if (symbol == EOF) {
            break;
        }
    }
    if (num_read) {
        text.push_back(AddWord(wrd, str_num, temp));
    }
}

void ParsePatt(vector <long long> &pattern) {
    long long symb;
    long long temp = 0;
    bool num_read = false;
    symb = getchar();
    while (true) {
        if (symb == ' ') {
            if (num_read) {
                pattern.push_back(temp);
                num_read = false;
                temp = 0;
            }
        } else {
            if (!num_read) {
                num_read = true;
            }
            temp = temp * 10 + (symb - '0');
        }
        symb = getchar();
        if (symb == '\n') {
            break;
        }
    }
    if (num_read) {
        pattern.push_back(temp);
    }
}

void ZFunction(vector <long long> &patt, vector <long long> &z) {
    long long right = 0, left = 0;
    long long size = patt.size();
    for (long long j = 1; j < size; ++j) {
        if (j <= right)
            z[j] = min(right - j + 1, z[j - left]);
        while ((j + z[j] < size) && (patt[size - 1 - z[j]] == patt[size - 1 - (j + z[j])])) {
            z[j]++;
        }
        if ((j + z[j] - 1) > right) {
            left = j;
            right = j + z[j] - 1;
        }
    }
}

void GoodSuff(vector <long long> &patt, vector <long long> &suff) {
    vector <long long> z(patt.size(), 0);
    ZFunction(patt, z);
    for (long long j = patt.size() - 1; j > 0; j--) {
        suff[patt.size() - z[j]] = j;
    }
    for (long long j = 1, r = 0; j < patt.size(); j++) {
        if (j + z[j] == patt.size()) {
            for (; r <= j; r++) {
                if (suff[r] == patt.size()) {
                    suff[r] = j;
                }
            }
        }
    }
}

void BadChar(vector <long long> &patt, map <long long, long long> &R) {
    for (long long i = patt.size() - 1; i >= 0; i--) {
        if (R.count(patt[i]) == 0) {
            R.insert(make_pair(patt[i], i));
        }
    }
}

void BoyerMoore(vector <long long> &patt, vector <TWords> &text) {
    vector <pair<long, long>> result;
    map <long long, long long> R;
    BadChar(patt, R);
    vector <long long> suff(patt.size() + 1, patt.size());
    GoodSuff(patt, suff);
    long long bound = 0;
    long long j;
    for (long long i = 0; i <= (long long)(text.size() - patt.size()); ) {
        for (j = patt.size() - 1; (j >= bound) && (patt[j] == text[i + j].val); j--) {} 
        //compare pattern and text until find unmatching
        if (j < bound) {
            result.push_back(make_pair(text[i].str, text[i].wrd));
            bound = patt.size() - suff[0];
            i += suff[0];
        } else {
            long long bad_suffix;
            if (R.count(text[i + j].val) == 1) {
                bad_suffix = R[text[i + j].val];
            } else {
                bad_suffix = -1;
            }
            i += max(suff[j + 1], j - bad_suffix); //use the rule of a good suffix or the rule of a bad char
            bound = 0;
        }
    }
    for (long long i = 0; i < result.size(); i++) {
        cout << result[i].first << ", " << result[i].second << '\n';
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    vector <long long> patt;
    vector <TWords> text;
    ParsePatt(patt);
    ParseText(text);
    BoyerMoore(patt, text);
    return 0;
}