#ifndef FIGURE_HPP
#define FIGURE_HPP
#include <iostream>
#include <memory>
#include <cmath>
#include <deque>
#include <vector>
#include <fstream>
#include <algorithm>

enum class FigureType {
    triangle,
    square,
    rectangle
};
// Перегрузка << 
template <class T>
std::ostream& operator <<(std::ostream& out, const std::pair<T,T> &my_pair) {
    out << "(" << my_pair.first << "," << my_pair.second << ")";
    return out; 
}


template <class T>
class Figure {
public:
    virtual T Area() = 0;
    virtual void Print() = 0;
    virtual std::pair<T,T> Centre() = 0;
    virtual void PrintInFile(std::ostream& pif) = 0;
};

template <class T>
class Triangle : public Figure<T> {
public:
    Triangle(): a(0,0),b(0,0),c(0,0), id("t")
    {}
    Triangle(T x1,T y1, T x2, T y2, T x3, T y3) : a(x1,y1),b(x2,y2),c(x3,y3),id("t")
    {
        T z1, z2, z3;
        z1 = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
	    z2 = sqrt(pow(x3 - x2, 2) + pow(y3 - y2, 2));
	    z3 = sqrt(pow(x1 - x3, 2) + pow(y1 - y3, 2));
	    if (z1 + z2 <= z3 || z2 + z3 <= z1 || z1 + z3 <= z2) {
		    throw std::logic_error("This is not triangle");
	    }
    }
    T Area() override { 
        return fabs(((a.first - c.first) * (b.second - c.second)
         - (b.first - c.first) * (a.second - c.second)) / 2);
    }
    std::pair<T,T> Centre() override {
        return std::make_pair((a.first + b.first + c.first) / 3,(a.second + b.second + c.second) / 3);
    }
    void Print() override {
        std::cout << "Triagle : ";
        std::cout << a << b << c << std::endl;
        std::cout << "Area : " << this->Area() << " Center : " << this->Centre() << std::endl;
        std::cout << "!End!" << std::endl;
    }

    void PrintInFile(std::ostream& pif) {
        pif << id << "\n";
        pif << a.first << " " << a.second << " ";
        pif << b.first << " " << b.second << " ";
        pif << c.first << " " << c.second << "\n";
        
    }
private:
    using vertex = std::pair<T,T>;
    vertex a, b, c;
    std::string id;
};


template <class T>
class Square : public Figure<T> {
public:
    Square() : a(0,0), b(0,0), id("s")
    {}
    Square(T x1, T y1, T x2, T y2) : a(x1,y1), b(x2,y2),id("s")
    {
        if((x1 - x2 != y1 - y2) || (x1 == x2) || (y1 == y2)) {
            throw std::logic_error("This are not opposite coordinates");
        }
    }
    T Area() override { 
        return (pow(b.first - a.first,2) + pow(b.second - a.second,2)) / 2;
    }
    std::pair<T,T>Centre() override {
        return std::make_pair((a.first + b.first) / 2,(a.second + b.second) / 2);
    }
    void Print() override {
        std::cout << "Square : ";
        std::cout << a << b << std::endl;
        std::cout << "Area : " << this->Area() << " Center : " << this->Centre() << std::endl;
        std::cout << "!End!" << std::endl;
    }

    void PrintInFile(std::ostream& pif) {
        pif << id << "\n";
        pif << a.first << " " << a.second << " ";
        pif << b.first << " " << b.second << "\n";   
    }
private:
    using vertex = std::pair<T,T>;
    vertex a, b;
    std::string id;
};


template <class T>
class Rectangle : public Figure<T> {
public:
    Rectangle() : a(0,0), b(0,0), id("s")
    {}
    Rectangle(T x1, T y1, T x2, T y2) : a(x1,y1), b(x2,y2),id("r")
    {}
    T Area() override { 
        return (pow(b.first - a.first,2) + pow(b.second - a.second,2)) / 2;
    }
    std::pair<T,T>Centre() override {
        return std::make_pair((a.first + b.first) / 2,(a.second + b.second) / 2);
    }
    void Print() override {
        std::cout << "Rectangle : ";
        std::cout << a << b << std::endl;
        std::cout << "Area : " << this->Area() << " Center : " << this->Centre() << std::endl;
        std::cout << "!End!" << std::endl;
    }
    void PrintInFile(std::ostream& pif) {
        pif << id << "\n";
        pif << a.first << " " << a.second << " ";
        pif << b.first << " " << b.second << "\n";   
    }
private:
    using vertex = std::pair<T,T>;
    vertex a, b;
    std::string id;
};

template <class T>
class FigureFactory {
public:
    virtual std::shared_ptr<Figure<T>> CreateFigure(std::istream& in) = 0;
};

template <class T>
class TriangleFactory : public FigureFactory<T> {
public:
    using ptr = std::shared_ptr<Figure<T>>;
    std::shared_ptr<Figure<T>> CreateFigure(std::istream& in) {
        T x1, y1, x2, y2, x3, y3;
        in >> x1; in >> y1; in >> x2; in >> y2;
        in >> x3; in >> y3;
        return ptr(std::make_shared<Triangle<T>>(x1,y1,x2,y2,x3,y3)); 
    }

};

template <class T>
class SquareFactory : public FigureFactory<T> {
public:
    using ptr = std::shared_ptr<Figure<T>>;
    std::shared_ptr<Figure<T>> CreateFigure(std::istream& in) {
        T x1, y1, x2, y2;
        in >> x1; in >> y1; in >> x2; in >> y2;
        return ptr(std::make_shared<Square<T>>(x1,y1,x2,y2));
    }
};


template <class T>
class RectangleFactory : public FigureFactory<T> {
public:
    using ptr = std::shared_ptr<Figure<T>>;
    std::shared_ptr<Figure<T>> CreateFigure(std::istream& in) {
        T x1, y1, x2, y2;
        in >> x1; in >> y1; in >> x2; in >> y2;
        return ptr(std::make_shared<Rectangle<T>>(x1,y1,x2,y2));
    }
};


#endif