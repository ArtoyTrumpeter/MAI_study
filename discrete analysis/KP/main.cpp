#include "source/map.hpp"
#include "source/search.hpp"


int main() {
    unsigned int n, m;
    std::cin >> n >> m;
    Map* testMap = new Map(n, m);
    unsigned int amount;
    std::cin >> amount;
    testMap->ConstructSymbolsMap(amount);
    Search* testSearch = new Search(testMap);
    std::cout << "Start(x,y) && Finish(x,y)" << std::endl;
    unsigned int x,y,x1,y1;
    std::cin >> x >> y >> x1 >> y1;
    auto start = std::make_pair(x, y);
    auto end = std::make_pair(x1, y1);
    testSearch->Result(start, end);
    testMap->Print();
    delete testMap;
    delete testSearch;
    return 0;
}