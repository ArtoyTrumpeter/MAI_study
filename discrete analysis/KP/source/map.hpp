#ifndef MAP_HPP
#define MAP_HPP
#include <iostream>
#include <vector>
#include <cmath>
#include <set>
#include <queue>


struct Information {
    std::string symbol; // 0 - true, # - false
    unsigned int allScore; // weight + heuristic(finish);
    unsigned int weight;
    std::pair<unsigned int, unsigned int> from; // parent_coordinate


    Information() {
        symbol = "0";
        allScore = 0;
        weight = 0;
        from = std::make_pair(0,0);
    }
};


class Map {
private:
    unsigned int lines;
    unsigned int colums;
    std::vector<std::vector<Information>> symbolsMap;
    std::set<std::pair<unsigned int, unsigned int>> noPoints;

public:
    friend class Search;

    Map(unsigned int lines, unsigned int colums) {
        this->lines = lines + 2;
        this->colums = colums + 2;
        symbolsMap.resize(this->lines);
    }


    void Print() {
        for(auto i = 0; i < lines;i++) {
            for(auto j = 0; j < colums;j++) {
                std::cout << symbolsMap[i][j].symbol << " ";
            }
            std::cout << std::endl;
        }
    }


    bool CheckNewPoint(int x, int y) {
    if(x > lines - 1 || y > colums - 1 || x <= 0 || y <= 0) {
        return false;
    }
    return true;
    }


    void ConstructSymbolsMap(unsigned int NoPointAmount) {
        std::string status;
        int temp = 0;
        for(auto i = 0; i < lines; i++) {
            for(auto j = 0; j < colums; j++) {
                symbolsMap[i].push_back(Information());
            }
        }
        for(auto j = 0; j < colums;j++) {
            symbolsMap[0][j].symbol = '#';
            symbolsMap[lines - 1][j].symbol = '#';
            noPoints.insert(std::make_pair(0, j));
            noPoints.insert(std::make_pair(lines - 1, j));
        }
        for(auto i = 0; i < lines;i++) {
            symbolsMap[i][0].symbol = '#';
            symbolsMap[i][colums - 1].symbol = '#';
            noPoints.insert(std::make_pair(i, 0));
            noPoints.insert(std::make_pair(i, colums - 1));
        }
        while(temp < NoPointAmount) {
            unsigned int pointX, pointY;
            std::cin >> pointX >> pointY;
            if(CheckNewPoint(pointX, pointY)) {
                symbolsMap[pointX][pointY].symbol = '#';
                noPoints.insert(std::make_pair(pointX, pointY));
                std::cout << "ADD" << std::endl;
            }
            temp++;
        }
        Print();
    }
};


#endif