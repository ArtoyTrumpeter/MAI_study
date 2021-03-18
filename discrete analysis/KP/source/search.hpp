#ifndef SEARCH_HPP
#define SEARCH_HPP
#include "map.hpp"
#include <map>
#include <algorithm>
#define const_weight 5 
using point = std::pair<unsigned int, unsigned int>;
using pointAndScore = std::pair<point, unsigned int>;


struct MyCompare {
    constexpr bool operator()(std::pair<point, unsigned int> const & a, std::pair<point, unsigned int> const & b) const noexcept {
        return a.second >= b.second;
    }
};


class Search {
public:
    Search(Map* map) {
        this->map = map;
    }

    void Result(point start, point finish) {
        if(map->CheckNewPoint(start.first,start.second) == true
            && map->CheckNewPoint(finish.first, finish.second) == true
            && AStar(start,finish) == true) {
                std::cout << "YES" << std::endl;
            ChangeSymbolMapAndPrintResult(start,finish);
        } else {
            std::cout << "ERROR" << std::endl;
        }
    }

private:
    Map* map;
    // открытый список
    std::priority_queue <pointAndScore,
            std::vector <pointAndScore>,MyCompare> openQueue;
    // множество точек в открытом списке
    std::set<point> pointsInOpenList;

    void ChangeSymbolMapAndPrintResult(point start, point finish) {
        point current = finish;
        int i = 1;
        while(current != start) {
            map->symbolsMap[current.first][current.second].symbol = '!';
            current = map->symbolsMap[current.first][current.second].from;
            i++;
        }
        map->symbolsMap[start.first][start.second].symbol = '!';
    }


    bool FindElement(std::set<point>& pointsInOpenList, point& old) {
        auto it = pointsInOpenList.find(old);
        if(it == pointsInOpenList.end()) {
            return false;
        }
        return true;
    }


    void Erase(std::set<point>& pointsInOpenList, point& old) {
        for(auto it = pointsInOpenList.begin();it != pointsInOpenList.end();it++) {
            if((*it) == old) {
                pointsInOpenList.erase(it);
                return;
            }
        }
    }


    unsigned int Heuristik(point start, point finish) {
        return ((abs(start.first - finish.first) + abs(start.second - finish.second)) * const_weight);
    }


    void GetFourVertex(std::vector<point>& adjacentVertex,point& current) {
        int i_const,j_const;
        i_const = current.first;
        j_const = current.second;
        int i_f,j_f,i_s,j_s;
        j_f = current.second - 1;
        j_s = current.second + 1;
        adjacentVertex.push_back({i_const,j_f});
        adjacentVertex.push_back({i_const,j_s});
        i_f = i_const - 1;
        i_s = i_const + 1;
        adjacentVertex.push_back({i_f, j_const});
        adjacentVertex.push_back({i_s, j_const});
    }


    void GetAllVertex(std::vector<point>& adjacentVertex,point& current) {
        for(auto j = current.second - 1; j <= current.second + 1; j++) {
            if(j == current.second) {
                unsigned int first = j - 1;
                unsigned int second = j + 1;
                adjacentVertex.push_back({current.first, first});
                adjacentVertex.push_back({current.first, second});
            }
            unsigned int high, low;
            high = current.first + 1;
            low = current.first - 1;
            adjacentVertex.push_back({low, j});
            adjacentVertex.push_back({high, j});
        }
    }


    bool AStar(point start, point finish) {
        openQueue.push({start,0});
        pointsInOpenList.insert(start);
        while(openQueue.size() > 0) {
            pointAndScore current = openQueue.top();
            if(current.first == finish) {
                return true;
            }
            openQueue.pop();
            Erase(pointsInOpenList, current.first);
            map->noPoints.insert(current.first);
            std::vector<point> adjacentVertex;
            unsigned int tenativeScore = 
                map->symbolsMap[current.first.first][current.first.second].weight + const_weight;
            GetFourVertex(adjacentVertex,current.first);
            for(size_t j = 0;j < adjacentVertex.size(); j++) {
                //Если клетка непроходима или находится в закрытом списке, игнорируем её.
                if(map->noPoints.find(adjacentVertex[j]) != map->noPoints.end()) {
                    continue;
                }
                /*
                Если клетка находится в открытом списке, то сравниваем её значение weight
                со значением weight таким, что если бы к ней пришли через текущую клетку.
                Если сохранённое в проверяемой клетке значение weight больше нового,
                то меняем её значение weight на новое, пересчитываем её значение allScore и
                изменяем указатель на родителя так, чтобы она указывала на текущую клетку.
                */
                if(FindElement(pointsInOpenList,adjacentVertex[j])) {
                    if(tenativeScore < map->symbolsMap[adjacentVertex[j].first][adjacentVertex[j].second].weight) {
                        map->symbolsMap[adjacentVertex[j].first][adjacentVertex[j].second].from = current.first;
                        map->symbolsMap[adjacentVertex[j].first][adjacentVertex[j].second].weight = tenativeScore;
                        map->symbolsMap[adjacentVertex[j].first][adjacentVertex[j].second].allScore = tenativeScore + Heuristik(adjacentVertex[j],finish); 
                    }
                //Если клетка не в открытом списке, то добавляем её в открытый список, при этом рассчитываем для неё значения, и также устанавливаем ссылку родителя на текущую клетку.
                } else {
                    map->symbolsMap[adjacentVertex[j].first][adjacentVertex[j].second].from = current.first;
                    map->symbolsMap[adjacentVertex[j].first][adjacentVertex[j].second].weight = tenativeScore;
                    map->symbolsMap[adjacentVertex[j].first][adjacentVertex[j].second].allScore = tenativeScore + Heuristik(adjacentVertex[j],finish);
                    //std::cout << adjacentVertex[j].first << " " << adjacentVertex[j].second << " " << map->symbolsMap[adjacentVertex[j].first][adjacentVertex[j].second].allScore << std::endl;
                    openQueue.push({adjacentVertex[j],map->symbolsMap[adjacentVertex[j].first][adjacentVertex[j].second].allScore});
                    pointsInOpenList.insert({adjacentVertex[j].first,adjacentVertex[j].second});
                }
            }
        }
        return false; 
    }
};

#endif