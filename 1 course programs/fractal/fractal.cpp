#include <iostream>
#include <vector>


using namespace std;


vector <string> f(vector <string>& a) 
{
    vector <string> r = a;
    int l = a[a.size()-1].length();
    string s; 
    s.resize(l, ' ');
    for(int i = 0; i < a.size(); i++) {
        r.push_back (a[i] + s.substr(0, l-2*i) + a[i]);
    }
    return r;
}


int main() 
{
    vector <string> r;
    r.push_back("+");
    int n;
    cin >> n;
    cout << endl;
    for(int i = 0; i < n; i++) {
        r = f(r);
    }
    for(int i = 0; i < r.size(); i++) {
        cout << r[i] << endl;
    }
}