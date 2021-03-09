#include "figure.hpp"

Point::Point() : x(0), y(0)
{}

Point::Point(double x, double y) : x(x), y(y)
{}

double Point::get_x() {
    return x;
}

double Point::get_y() {
    return y;
}

void Point::print_point() {
    std::cout << "Point : (" << get_x() << ", " << get_y() << ")" << std::endl;
}