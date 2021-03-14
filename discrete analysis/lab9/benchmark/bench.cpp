#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

struct TNode {
	int id;
	char color = 'w';
	vector<int> edges;
};

void DfsVisit(vector<TNode> &G, TNode &u, vector<int> &answ) {
	u.color = 'g';
	for (int i = 0; i < u.edges.size(); i++) {
		if (G[u.edges[i]].color == 'w') {
			answ.push_back(u.edges[i]);
			DfsVisit(G, G[u.edges[i]], answ);
		}
	}
	u.color = 'b';
}

void DFS(vector<TNode> &G) {
	vector<int> answ;
	for (int i = 1; i < G.size(); i++) {
		if (G[i].color == 'w') {
			answ.push_back(G[i].id);
			DfsVisit(G, G[i], answ);
			sort(answ.begin(), answ.end());
			/*for (int j = 0; j < answ.size(); j++) {
				cout << answ[j];
				j == answ.size() - 1 ? cout << '\n' : cout << ' ';
			}*/
			answ.clear();
		}
	}
	
}

int main () {
	int n, m, from, to;
    double start, end;
	cin >> n >> m;
	vector<TNode> G(n + 1);
	for (int i = 1; i <= n; i++) {
		G[i].id = i;
	}
	for (int i = 0; i < m; i++) {
		cin >> from >> to;	
		G[from].edges.push_back(to);
		G[to].edges.push_back(from);
	}
    start = clock();
	DFS(G);
    end = clock();
    cout << "Time of working " << (end - start) / CLOCKS_PER_SEC << "sec" << std::endl;
	return 0;
}