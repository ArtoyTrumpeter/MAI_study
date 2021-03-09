#include "square.hpp"

Square::Square() : p1(0, 0), p2(0, 0) {}

Square::Square(Point point1, Point point2) : 
    p1(point1), p2(point2) {
    if (((p1.get_x() - p2.get_x()) != (p1.get_y() - p2.get_y())) && 
    ((p1.get_x() - p2.get_x()) != (-(p1.get_y() - p2.get_y()))) ||
    ((p1.get_x() == p2.get_x()) && (p1.get_y() == p2.get_y()))) {
        throw std::logic_error("This is not a square!");
    }
}

void Square::print() {
    std::cout << "Square tops" << std::endl;
    Point(p1.get_x(), p1.get_y()).print_point();
    Point(p2.get_x(), p1.get_y()).print_point();
    Point(p2.get_x(), p2.get_y()).print_point();
    Point(p1.get_x(), p2.get_y()).print_point();
}

double Square::get_area() {
    return (pow((p1.get_x() - p2.get_x()),2) + pow((p1.get_y() - p2.get_y()),2)) / 2;
}

Point Square::get_center() {
    return Point((p1.get_x() + p2.get_x()) / 2, (p1.get_y() + p2.get_y()) / 2);
}