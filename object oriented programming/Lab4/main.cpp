/* Тояков Артем Олегович М8о-207Б-18
Разработать шаблоны классов согласно варианту задания.
Параметром шаблона должен являться скалярный тип данных задающий тип данных для оси координат.
Классы должны иметь публичные поля. Фигуры являются фигурами вращения.
Для хранения координат фигур необходимо использовать шаблон  std::pair.
Создать набор шаблонов, создающих функции, реализующие:
1.       Вычисление геометрического центра фигуры;
2.       Вывод в стандартный поток вывода std::cout координат вершин фигуры;
3.       Вычисление площади фигуры;
Вариант 25:
Треугольник, квадрат, прямоугольник.
*/

#include "figure.hpp"


void help() {
    std::cout << "Press s to show information about square" << std::endl;
    std::cout << "Press t to show information about triangle" << std::endl;
    std::cout << "Press r to show information about rectangle" << std::endl;
    std::cout << "Then enter tuple to use std::tuple<> or class to use class" << std::endl;
}


int main() {
    help();
    using point = std::pair<double, double>;
	using std::make_pair;
    std::cout << std::fixed;
    std::cout.precision(3);
    std::string data, s;
    while(std::cin >> data) {
        if(data == "r") {
            double x1, y1, x2, y2, x3, y3, x4, y4;
            std::cout << "Coordinates :" << std::endl;
            std::cin >> x1 >> y1 >> x2 >> y2 >> x3 >> y3 >> x4 >> y4;
            if(check(x1, y1, x2, y2, x3, y3, x4, y4) == true) {
                std::cout << "tuple or class" << std::endl;
                std::cin >> s;
                if(s == "tuple") {
                    std::tuple<point, point, point, point>  rectangle(make_pair(x1, y1), make_pair(x2, y2), 
                        make_pair(x3, y3), make_pair(x4, y4));
                    information(rectangle);
                } else if(s == "class") {
                    Rectangle <double> rectangle(x1, y1, x2, y2, x3, y3, x4, y4);
                    information(rectangle);
                }
            }
        } else if(data == "s") {
            double x1, y1, x2, y2;
            std::cout << "Coordinates :" << std::endl;
            std::cin >> x1 >> y1 >> x2 >> y2;
            if(check(x1, y1, x2, y2) == true) {
                std::cout << "tuple or class" << std::endl;
                std::cin >> s;
                if(s == "tuple") {
                    std::tuple<point, point>  square(make_pair(x1, y1), make_pair(x2,y2));
                    information(square);
                } else if(s == "class") {
                    Square <double> square(x1,y1,x2,y2);
                    information(square);
                }
            }
        } else if(data == "t") {
            double x1, y1, x2, y2, x3, y3;
            std::cout << "Coordinates :" << std::endl;
            std::cin >> x1 >> y1 >> x2 >> y2 >> x3 >> y3;
            if(check(x1, y1, x2, y2, x3, y3) == true) {
                std::cout << "tuple or class" << std::endl;
                std::cin >> s;
                if(s == "tuple") {
                    std::tuple<point, point, point>  triangle(make_pair(x1, y1), make_pair(x2,y2),make_pair(x3,y3));
                    information(triangle);
                } else if(s == "class") {
                    Triangle <double> triangle(x1, y1, x2, y2, x3, y3);
                    information(triangle);
                }
            }
        } else if(data == "exit") {
            return 0;
        }
    }
    return 0;
}