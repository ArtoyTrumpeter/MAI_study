#ifndef _SQUARE_
#define _SQUARE_

#include "figure.hpp"

class Square: public Figure {
private:
    Point p1, p2;
public:
    Square();
    Square(Point point1, Point point2);
    void print() override;
    double get_area() override;
    Point get_center() override;
};

#endif //_SQUARE_