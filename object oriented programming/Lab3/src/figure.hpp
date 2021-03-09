#ifndef _FIGURE_POINT_
#define _FIGURE_POINT_

#include <iostream>
#include <math.h>
#include <vector>

class Point
{
private:
    double x, y;
public:
    Point();
    Point(double x, double y);
    double get_x();
    double get_y();
    void print_point();
};

class Figure
{
public:
    virtual void print() = 0;
    virtual double get_area() = 0;
    virtual Point get_center() = 0;
};

#endif //_FIGURE_POINT