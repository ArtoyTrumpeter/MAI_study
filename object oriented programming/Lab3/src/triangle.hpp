#ifndef _TRIANGLE_
#define _TRIANGLE_

#include "figure.hpp"

class Triangle : public Figure {
private:
    Point p1, p2, p3;
public:
    Triangle();
    Triangle(Point point1, Point point2, Point point3);
    void print() override;
    double get_area() override;
    Point get_center() override;
};

#endif //_TRIANGLE_