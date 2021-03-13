#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

struct TSegment {
	int left;
	int right;
	int ind;	
};

bool Cmp (TSegment &a, TSegment &b) {
	return a.ind < b.ind;
}

void Selection(vector<TSegment> &segs, int m) {
	vector<TSegment> answer;
	TSegment begin;
	begin.left = 0;
	begin.right = 0;
	answer.push_back(begin);
	while (answer.back().right < m) {
		int max = 0;
		int index = -1;
		for (int i = 0; i < segs.size(); i++) {
			if (segs[i].left <= answer.back().right && segs[i].right > answer.back().right) {
				if (segs[i].right > max) {
					max = segs[i].right;
					index = i;
				}
			}
		}
		if (index == -1) {
			cout << "0\n";
			return;
		} else {
			answer.push_back(segs[index]);
		}
	}

	/*sort(answer.begin() + 1, answer.end(), Cmp);
	cout << answer.size() - 1 << '\n';
	for (int i = 1; i < answer.size(); i++) {
		cout << answer[i].left << " " << answer[i].right << '\n';
	}*/
	return;

}

int main() {
	int n, m;
    double start, end;
	cin >> n;
	vector<TSegment> segs(n);
	for (int i = 0; i < n; i++) {
		segs[i].ind = i;
		cin >> segs[i].left >> segs[i].right; 
	}
	cin >> m;
    start = clock();
	Selection(segs, m);
    end = clock();
    std::cout << "Time of working " << (end - start) / CLOCKS_PER_SEC << "sec" << std::endl;
	return 0;
}