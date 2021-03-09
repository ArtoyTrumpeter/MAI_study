#ifndef _FIGURE_
#define _FIGURE_

#include <iostream>
#include <tuple>
#include <cmath>

template <class T>
struct Triangle {
    using type = T;
    using my_pare = std::pair<T,T>;
    my_pare a, b, c;
    Triangle(T x1, T y1, T x2, T y2, T x3, T y3) : a(x1,y1), b(x2,y2), c(x3,y3)
    {}
};

template <class T>
struct  Square {
    using type = T;
    using my_pare = std::pair<T,T>;
    my_pare a, b;    
    Square(T x1, T y1, T x2, T y2) : a(x1,y1), b(x2,y2)
    {}
};

template <class T>
struct Rectangle {
    using type = T;
    using my_pair = std::pair<T,T>;
    my_pair a, b, c, d;
    Rectangle(T x1, T y1, T x2, T y2, T x3, T y3, T x4, T y4) : a(x1,y1), b(x2,y2), c(x3,y3), d(x4, y4)
    {}
};

template <class T>
bool check(T x1, T y1, T x2, T y2, T x3, T y3) {
        T a1, a2, a3;
        a1 = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
        a2 = sqrt(pow(x3 - x2, 2) + pow(y3 - y2, 2));
        a3 = sqrt(pow(x1 - x3, 2) + pow(y1 - y3, 2));
	    if ((a1 + a2 <= a3) || (a2 + a3 <= a1) || (a1 + a3 <= a2)) {
            std::cout << "Error" << std::endl;
		    return false;
	    } else {
            return true;
        }
    }

template <class T>
bool check(T x1, T y1, T x2, T y2) {
    if((x2 - x1 != y2 - y1) || (x2 == x1 && y2 == y1)) {
        std::cout << "Error" << std::endl;
        return false;
    } else {
        return true;
    }
}

template <class T>
bool check(T x1, T y1, T x2, T y2, T x3, T y3, T x4, T y4) {
    T a1, a2, a3, a4, diag;
    a1 = pow(x2 - x1, 2) + pow(y2 - y1, 2);
    a2 = pow(x3 - x2, 2) + pow(y3 - y2, 2);
    a3 = pow(x4 - x3, 2) + pow(y4 - y3, 2);
    a4 = pow(x1 - x4, 2) + pow(y1 - y4, 2);
    diag = pow(x3 - x1, 2) + pow(y3 - y1, 2);
    if (((a1 + a2) != diag) || ((a3 + a4) != diag)) {
        std::cout << "Error" << std::endl;
        return false;
    } else {
        return true;
    }
}

template <template <class> class F, class T>
typename std::enable_if< std::is_same< F<T>, Triangle<T> >::value, F<T> >::type information(F<T> t) {
    std::cout << "Triangle :" << std::endl;
    T sq, x1, y1, x2, y2, x3, y3;
    x1 = (t.a.first + t.b.first + t.c.first) / 3;
    y1 = (t.a.second + t.b.second + t.c.second) / 3;
    std::cout << "Center point (" << x1 << "," << y1 << ")" << std::endl;
    x1 = t.a.first; y1 = t.a.second; 
    x2 = t.b.first; y2 = t.b.second;
    x3 = t.c.first; y3 = t.c.second;
    sq = fabs(((x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)) / 2);
    std::cout << "Area : " << sq << std::endl;
    std::cout << "Coordinates :" << std::endl;
    std::cout << "(" << x1 << "," << y1 << ")" << std::endl;
    std::cout << "(" << x2 << "," << y2 << ")" << std::endl;
    std::cout << "(" << x3 << "," << y3 << ")" << std::endl;
    return t;
}

template <template <class> class F, class T>
typename std::enable_if< std::is_same< F<T>, Square<T> >::value, F<T> >::type information(F<T> s) {
    std::cout << "Square :" << std::endl;
    T x, y, sq;
    x = (s.a.first + s.b.first) / 2;
    y = (s.b.second + s.a.second) / 2;
    std::cout << "Center point (" << x << "," << y << ")" << std::endl;
    x = (s.b.first - s.a.first);
    y = (s.b.second - s.a.second);
    sq = (pow(x,2) + pow(y,2)) / 2;
    std::cout << "Area : " << sq << std::endl;
    std::cout << "Coordinates :" << std::endl;
    x = s.a.first;
    y = s.a.second;
    std::cout << "(" << x << "," << y << ")" << std::endl;
    x = s.a.first;
    y = s.b.second;
    std::cout << "(" << x << "," << y << ")" << std::endl;
    x = s.b.first;
    y = s.b.second;
    std::cout << "(" << x << "," << y << ")" << std::endl;
    x = s.b.first;
    y = s.a.second;
    std::cout << "(" << x << "," << y << ")" << std::endl;
    return s;
}

template <template <class> class F, class T>
typename std::enable_if< std::is_same< F<T>, Rectangle<T> >::value, F<T> >::type information(F<T> r) {
    std::cout << "Rectangle :" << std::endl;
    T sq, x1, y1, x2, y2, x3, y3, x4, y4;
    x1 = (r.a.first + r.c.first) / 2;
    y1 = (r.a.second + r.c.second) / 2;
    std::cout << "Center point (" << x1 << "," << y1 << ")" << std::endl;
    x1 = sqrt(pow(r.a.first - r.b.first, 2) + pow(r.a.second - r.b.second, 2));
    x2 = sqrt(pow(r.b.first - r.c.first, 2) + pow(r.b.second - r.c.second, 2));
    sq = x1 * x2;
    x1 = r.a.first; y1 = r.a.second; 
    x2 = r.b.first; y2 = r.b.second;
    x3 = r.c.first; y3 = r.c.second;
    x4 = r.d.first; y4 = r.d.second;
    std::cout << "Area : " << sq << std::endl;
    std::cout << "Coordinates :" << std::endl;
    std::cout << "(" << x1 << "," << y1 << ")" << std::endl;
    std::cout << "(" << x2 << "," << y2 << ")" << std::endl;
    std::cout << "(" << x3 << "," << y3 << ")" << std::endl;
    std::cout << "(" << x4 << "," << y4 << ")" << std::endl;    
    return r;
}

template <class T>
void information(std::tuple<std::pair<T,T>, std::pair<T,T>, std::pair<T,T>> Triangle) {
    std::cout << "Triangle :" << std::endl;
    T x1, x2, x3, y1, y2, y3, sq;
    x1 = (std::get<0>(Triangle).first + std::get<1>(Triangle).first + std::get<2>(Triangle).first) / 3;
    y1 = (std::get<0>(Triangle).second + std::get<1>(Triangle).second + std::get<2>(Triangle).second) / 3;
    std::cout << "Center point (" << x1 << "," << y1 << ")" << std::endl;
    x1 = std::get<0>(Triangle).first; y1 = std::get<0>(Triangle).second; 
    x2 = std::get<1>(Triangle).first; y2 = std::get<1>(Triangle).second;
    x3 = std::get<2>(Triangle).first; y3 = std::get<2>(Triangle).second;
    sq = fabs(((x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)) / 2);
    std::cout << "Area : " << sq << std::endl;
    std::cout << "Coordinates :" << std::endl;
    std::cout << "(" << x1 << "," << y1 << ")" << std::endl;
    std::cout << "(" << x2 << "," << y2 << ")" << std::endl;
    std::cout << "(" << x3 << "," << y3 << ")" << std::endl;
}

template <class T>
void information(std::tuple<std::pair<T,T>, std::pair<T,T>> Square) {
    std::cout << "Square :" << std::endl;
    T cx, cy, sq;
    cx = (std::get<0>(Square).first + std::get<1>(Square).first) / 2;
    cy = (std::get<0>(Square).second + std::get<1>(Square).second) / 2;
    std::cout << "Center point (" << cx << "," << cy << ")" << std::endl;
    cy = std::get<1>(Square).second - std::get<0>(Square).second;
    cx = std::get<1>(Square).first - std::get<0>(Square).first; 
    sq = (pow(cx,2) + pow(cy,2)) / 2;
    std::cout << "Area : " << sq << std::endl;
    std::cout << "Coordinates :" << std::endl;
    cx = std::get<0>(Square).first;
    cy = std::get<0>(Square).second;
    std::cout << "(" << cx << "," << cy << ")" << std::endl;
    cx = std::get<0>(Square).first;
    cy = std::get<1>(Square).second;
    std::cout << "(" << cx << "," << cy << ")" << std::endl;
    cx = std::get<1>(Square).first;
    cy = std::get<1>(Square).second;
    std::cout << "(" << cx << "," << cy << ")" << std::endl;
    cx = std::get<1>(Square).first;
    cy = std::get<0>(Square).second;
    std::cout << "(" << cx << "," << cy << ")" << std::endl;
}

template <class T>
void information(std::tuple<std::pair<T,T>, std::pair<T,T>, std::pair<T,T>, std::pair<T,T>> Rectangle) {
    std::cout << "Rectangle :" << std::endl;
    T x1, x2, x3, x4, y1, y2, y3, y4, sq;
    x1 = (std::get<0>(Rectangle).first + std::get<2>(Rectangle).first) / 2;
    y1 = (std::get<0>(Rectangle).second + std::get<2>(Rectangle).second) / 2;
    std::cout << "Center point (" << x1 << "," << y1 << ")" << std::endl;
    x1 = sqrt(pow(std::get<1>(Rectangle).first - std::get<0>(Rectangle).first, 2) + 
        pow(std::get<1>(Rectangle).second - std::get<0>(Rectangle).second, 2));
    x2 = sqrt(pow(std::get<2>(Rectangle).first - std::get<1>(Rectangle).first, 2) + 
        pow(std::get<2>(Rectangle).second - std::get<1>(Rectangle).second, 2));
    sq = x1 * x2;
    x1 = std::get<0>(Rectangle).first; y1 = std::get<0>(Rectangle).second; 
    x2 = std::get<1>(Rectangle).first; y2 = std::get<1>(Rectangle).second;
    x3 = std::get<2>(Rectangle).first; y3 = std::get<2>(Rectangle).second;
    x4 = std::get<3>(Rectangle).first; y4 = std::get<3>(Rectangle).second;
    std::cout << "Area : " << sq << std::endl;
    std::cout << "Coordinates :" << std::endl;
    std::cout << "(" << x1 << "," << y1 << ")" << std::endl;
    std::cout << "(" << x2 << "," << y2 << ")" << std::endl;
    std::cout << "(" << x3 << "," << y3 << ")" << std::endl;
    std::cout << "(" << x4 << "," << y4 << ")" << std::endl;
}

#endif //_FIGURE_