#include "rectangle.hpp"

Rectangle::Rectangle() : p1(0, 0), p2(0, 0), p3(0, 0), p4(0, 0){}

Rectangle::Rectangle(Point point1, Point point2, Point point3, Point point4) : 
    p1(point1), p2(point2), p3(point3), p4(point4){
    double diag, x1, x2, x3, x4;
    diag = pow(p3.get_x() - p1.get_x(), 2) + pow(p3.get_y() - p1.get_y(), 2);
    x1 = pow(p2.get_x() - p1.get_x(), 2) + pow(p2.get_y() - p1.get_y(), 2);
    x2 = pow(p3.get_x() - p2.get_x(), 2) + pow(p3.get_y() - p2.get_y(), 2);
    x3 = pow(p4.get_x() - p3.get_x(), 2) + pow(p4.get_y() - p3.get_y(), 2);
    x4 = pow(p1.get_x() - p4.get_x(), 2) + pow(p1.get_y() - p4.get_y(), 2);
    if (((x1 + x2) != diag) || ((x3 + x4) != diag)) {
        throw std::logic_error("This is not a Rectangle!");
    }
}

void Rectangle::print() {
    std::cout << "Rectangle tops" << std::endl;
    Point(p1.get_x(), p1.get_y()).print_point();
    Point(p2.get_x(), p2.get_y()).print_point();
    Point(p3.get_x(), p3.get_y()).print_point();
    Point(p4.get_x(), p4.get_y()).print_point();
}

double Rectangle::get_area() {
    double x1, x2;
    x1 = sqrt(pow(p2.get_x() - p1.get_x(), 2) + pow(p2.get_y() - p1.get_y(), 2));
    x2 = sqrt(pow(p3.get_x() - p2.get_x(), 2) + pow(p3.get_y() - p2.get_y(), 2));
    return x1 * x2;
}

Point Rectangle::get_center() {
    return Point((p1.get_x() + p3.get_x()) / 2, (p1.get_y() + p3.get_y()) / 2);
}