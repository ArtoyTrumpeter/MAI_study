/*Тояков Артем Олегович М8о-207Б-18
Разработать классы согласно варианту задания, классы должны наследоваться от базового класса Figure. 
Фигуры являются фигурами вращения. Все классы должны поддерживать набор общих методов:
1.   	Вычисление геометрического центра фигуры;
2.   	Вывод в стандартный поток вывода std::cout координат вершин фигуры; 
3.      Вычисление площади фигуры;
Фигуры треугольник, квадрат, прямоугольник.
Создать программу, которая позволяет:
•       Вводить из стандартного ввода std::cin фигуры, согласно варианту задания.
•       Сохранять созданные фигуры в динамический массив std::vector<Figure*>
•   	Вызывать для всего массива общие функции (1-3 см. выше), т.е. 
*           распечатывать для каждой фигуры в массиве геометрический центр, координаты вершин и площадь.
•       Необходимо уметь вычислять общую площадь фигур в массиве.
•       Удалять из массива фигуру по индексу;
*/

#include "src/figure.hpp"
#include "src/triangle.hpp"
#include "src/square.hpp"
#include "src/rectangle.hpp"

using namespace std;

void help() {
    cout << "add = enter figure's coordinates and add her in vector" << endl;
    cout << "print_all = show information about all figures" << endl;
    cout << "all_area = the sum area of all figures" << endl;
    cout << "delete = delete figure by index" << endl;
    cout << "print = show information about figure by index" << endl;
    cout << "size = the size of our array of figures" << endl;
    cout << "help = helping list" << endl;
    cout << "exit = exit" << endl;
}

void add(vector<Figure*> & figures) {
    cout << "Press t to add triangle, s to add square, r to add rectangle" << endl;
    string your_figure;
    Triangle *t = nullptr;
    Square *s = nullptr;
    Rectangle *r = nullptr;
    cin >> your_figure;
    if (your_figure == "t") {
        cout << "Please, enter coordinates of three vertices" << endl;
        double x1, y1, x2, y2, x3, y3;
        cin >> x1 >> y1 >> x2 >> y2 >> x3 >> y3;
        Point p1(x1, y1), p2(x2, y2), p3(x3, y3);
        try {
			t = new Triangle(p1, p2, p3);
            figures.push_back(dynamic_cast<Figure*>(t));
		} catch (logic_error & err) {
            delete t;
			cout << err.what() << endl;
		}
    } else if (your_figure == "s") {
        cout << "Please, enter coordinates of two opposite vertices" << endl;
        double x1, y1, x2, y2;
        cin >> x1 >> y1 >> x2 >> y2;
        Point p1(x1, y1), p2(x2, y2);
        try {
			s = new Square(p1, p2);
            figures.push_back(dynamic_cast<Figure*>(s));
		} catch (logic_error & err) {
            delete s;
			cout << err.what() << endl;
		}
    } else if (your_figure == "r"){
        cout << "Please, enter four ordered vertices" << endl;
        double x1, y1, x2, y2, x3, y3, x4, y4;
        cin >> x1 >> y1 >> x2 >> y2 >> x3 >> y3 >> x4 >> y4;
        Point p1(x1, y1), p2(x2, y2), p3(x3, y3), p4(x4, y4);
        try {
			r = new Rectangle(p1, p2, p3, p4);
            figures.push_back(dynamic_cast<Figure*>(r));
		} catch (logic_error & err) {
            delete r;
			cout << err.what() << endl;
		}
    } else {
        return;
    }
}

vector<Figure*> delete_el(vector<Figure*> & figures, int del_index) {
    vector<Figure*> new_figures;
    for (int i = 0; i < figures.size(); i++) {
        if (i != del_index) {
            new_figures.push_back(figures[i]);
        }
    }
    figures.clear();
    return new_figures;   
}

int main() {
    cout << fixed;
    cout.precision(3);
    vector <Figure*> figures;
    string option;
    help();
    while (cin >> option) {
        if (option == "add") {
            add(figures);
        } else if (option == "delete") { 
            cout << "Index = ";
            int index;
            cin >> index;
            if ((index < 0) || (index >= figures.size())) {
                throw logic_error("Incorrect index");
            }
            try {
               delete figures[index];
               figures = delete_el(figures, index);
            } catch(logic_error & err) {
                cout << err.what() << endl;
            }
        } else if(option == "print") {
            cout << "Index = ";
            int index;
            cin >> index;
            if ((index < 0) || (index >= figures.size())) {
                throw logic_error("Incorrect index");
            }
            try {
                figures[index]->print();
                cout << "center ";
                figures[index]->get_center().print_point();
                cout << "Area = " << figures[index]->get_area() << endl;
            } catch(logic_error & err) {
                cout << err.what() << endl;
            }
        } else if (option == "print_all") {
            for(int j = 0; j < figures.size(); j++) {
                cout << j << "-st figure" << endl;
                figures[j]->print();
                cout << "center ";
                figures[j]->get_center().print_point();
                cout << "Area = " << figures[j]->get_area() << endl;
            }
        } else if (option == "size") {
            cout << figures.size() << endl;
        } else if (option == "all_area") {
            double summary_area = 0;
            for(int j = 0; j < figures.size(); j++) {
                summary_area = summary_area + figures[j]->get_area();
            }
            cout << "Area of all figures = " << summary_area << endl;
        } else if (option == "exit") {
            for(int j = 0; j < figures.size(); j++) {
                delete figures[j];
            }
            return 0;
        } else if (option == "help") {
            help();
        }
    }
    return 0;
}