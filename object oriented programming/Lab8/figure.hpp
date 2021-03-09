#ifndef FIGURE_HPP
#define FIGURE_HPP
#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <queue>
#include <fstream>
#include <unordered_map>
#include <algorithm>

// Перегрузка << 
template <class T>
std::ostream& operator <<(std::ostream& out, const std::pair<T,T> &my_pair) {
    out << "(" << my_pair.first << "," << my_pair.second << ")";
    return out; 
}


template <class T>
class Figure {
public:
    virtual std::ostream& Print(std::ostream& out) = 0;
};

template <class T>
class Triangle : public Figure<T> {
private:
    using type = T;
    using vertex_t = std::pair<T,T>;
    vertex_t a,b,c;
public:
    Triangle() : a(0,0), b(0,0), c(0,0)
    {}
    Triangle(T x1, T y1, T x2, T y2, T x3, T y3) : a(x1,y1), b(x2,y2), c(x3,y3)
    {
        T z1, z2, z3;
        z1 = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
	    z2 = sqrt(pow(x3 - x2, 2) + pow(y3 - y2, 2));
	    z3 = sqrt(pow(x1 - x3, 2) + pow(y1 - y3, 2));
	    if (z1 + z2 <= z3 || z2 + z3 <= z1 || z1 + z3 <= z2) {
		    throw std::logic_error("This is not triangle");
	    }
    }

    std::ostream& Print(std::ostream& out) override {
        out << "Triangle : " << a << "," << b << "," << c << "\n";
        return out;
    }
};

template <class T>
class Square : public Figure<T> {
private:
    using type = T;
    using vertex_t = std::pair<T,T>;
    vertex_t a,b;
public:
    Square() : a(0,0), b(0,0)
    {}

    Square(T x1, T y1, T x2, T y2) : a(x1,y1), b(x2,y2)
    {
        if((x1 - x2 != y1 - y2) || (x1 == x2) || (y1 == y2)) {
            throw std::logic_error("These are not opposite coordinates");
        }
    }

    std::ostream& Print(std::ostream& out) override {
        out << "Square : " << a << "," << b << "\n";
        return out;
    }
};

template <class T>
class Rectangle : public Figure<T> {
private:
    using type = T;
    using vertex_t = std::pair<T,T>;
    vertex_t a,b;
public:
    Rectangle() : a(0,0), b(0,0)
    {}

    Rectangle(T x1, T y1, T x2, T y2) : a(x1,y1), b(x2,y2)
    {}

    std::ostream& Print(std::ostream& out) override {
        out << "Rectangle : " << a << "," << b << "\n";
        return out;
    }
};

#endif