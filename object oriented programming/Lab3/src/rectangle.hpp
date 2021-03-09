#ifndef _RECTANGLE_
#define _RECTANGLE_

#include "figure.hpp"

class Rectangle: public Figure {
private:
    Point p1, p2, p3, p4;
public:
    Rectangle();
    Rectangle(Point point1, Point point2, Point point3, Point point4);
    void print() override;
    double get_area() override;
    Point get_center() override;
};

#endif //_RECTANGLE_