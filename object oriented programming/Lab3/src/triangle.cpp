#include "triangle.hpp"

Triangle::Triangle() : p1(0, 0), p2(0, 0), p3(0, 0)
{}

Triangle::Triangle(Point point1, Point point2, Point point3) : p1(point1), p2(point2), p3(point3) {
    double x1, x2, x3;
    x1 = sqrt(pow(p2.get_x() - p1.get_x(), 2) + (pow(p2.get_y() - p1.get_y(), 2)));
    x2 = sqrt(pow(p3.get_x() - p2.get_x(), 2) + (pow(p3.get_y() - p2.get_y(), 2)));
    x3 = sqrt(pow(p1.get_x() - p3.get_x(), 2) + (pow(p1.get_y() - p3.get_y(), 2)));
    if (((x1 + x2) <= x3) || ((x2 + x3) <= x1) || (x1 + x3 <= x2)) {
        throw std::logic_error("This is not triangle");
    }
}

void Triangle::print() {
    std::cout << "Triangle tops" << std::endl;
    Point(p1.get_x(), p1.get_y()).print_point();
    Point(p2.get_x(), p2.get_y()).print_point();
    Point(p3.get_x(), p3.get_y()).print_point();
}

double Triangle::get_area() {
    return fabs(((p1.get_x() - p3.get_x()) * (p2.get_y() - p3.get_y()) - 
        (p2.get_x() - p3.get_x()) * (p1.get_y() - p3.get_y())) / 2);
}

Point Triangle::get_center() {
    return Point(((p1.get_x() + p2.get_x() + p3.get_x()) / 3), ((p1.get_y() + p2.get_y() + p3.get_x()) / 3));
}