//класс фигуры квадрат

#ifndef SQUARE_HPP
#define SQUARE_HPP
#include <iostream>
#include <cmath>
#include <algorithm>
#include <memory>
#include <cassert>

template <class T>
struct Square {
    using type = T;
    using vertex_t = std::pair<T,T>;
    vertex_t a,b;
    Square(T x1, T y1,T x2, T y2) :  a(x1,y1),  b(x2,y2)
    {
        if((x2 - x1 != y2 - y1) || (x2 == x1 && y2 == y1)) {
            throw std::logic_error("This is not square");
        }
    }
    ~Square() {
        a.first = 0;
        a.second = 0;
        b.first = 0;
        b.second = 0;
    }
};


template <template <class> class F, class T>
typename std::enable_if< std::is_same< F<T>, Square<T> >::value, F<T> >::type information(const F<T>& s) noexcept {
    std::cout << "Square :" << std::endl;
    T x,y,sq;
    x = (s.a.first + s.b.first) / 2;
    y = (s.b.second + s.a.second) / 2;
    std::cout << "Center point (" << x << "," << y << ") ";
    x = (s.b.first - s.a.first);
    y = (s.b.second - s.a.second);
    sq = (pow(x,2) + pow(y,2)) / 2;
    std::cout << "Area : " << sq << std::endl;
    std::cout << "Coordinates :" << std::endl;
    x = s.a.first;
    y = s.a.second;
    std::cout << "(" << x << "," << y << ")  ";
    x = s.a.first;
    y = s.b.second;
    std::cout << "(" << x << "," << y << ")  ";
    x = s.b.first;
    y = s.b.second;
    std::cout << "(" << x << "," << y << ")  ";
    x = s.b.first;
    y = s.a.second;
    std::cout << "(" << x << "," << y << ")" << std::endl;
    return s;
}

#endif