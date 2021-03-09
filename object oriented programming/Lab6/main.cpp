/*
Тояков Артем M8o-207Б-18,вариант 28 вектор, очередь, квадрат.
Реализовать программу, которая:
1)Позволяет вводить с клавиатуры фигуры (с типом int в качестве параметра шаблона фигуры) и добавлять в коллекцию;
2)Позволяет удалять элемент из коллекции по номеру элемента;
3)Выводит на экран введенные фигуры и выводит на экран количество объектов, у которых площадь меньше заданной  c помощью std::for_each;
*/


#include "square.hpp"
#include "queue.hpp"
#include "allocator.hpp"
#include "vector.hpp"


void help() {
    std::cout << "add = add figure" << std::endl;
    std::cout << "del = delete figure" << std::endl;
    std::cout << "size = size of vector" << std::endl;
    std::cout << "print_all = show all figures in queue" << std::endl;
    std::cout << "exit = exit" << std::endl;
}

int main() {
    help();
    std::string inf;
    TVector<Square<double>, TAllocator<Square<double>,5>> vktr(2);
    while(std::cin >> inf) {
        if(inf == "add") {
            try {
                double x,y,x1,y1;
                std::cout << "Your x1 y1 x2 y2 : ";
                std::cin >> x >> y >> x1 >> y1;
                Square<double> o(x,y,x1,y1);
                vktr.PushBack(o);
			} catch (std::logic_error& err) {
				std::cout << err.what() << std::endl;
			}
        } else if (inf == "del") {
            try {
                int a;
                std::cout << "Position ";
                std::cin >> a;
                vktr.Erase(a);
			} catch (std::logic_error& err) {
				std::cout << err.what() << std::endl;
			}
        } else if (inf == "exit") {
            return 0;
        } else if(inf == "print_all") {
            std::for_each(vktr.begin(),vktr.end(),[](Square<double> o)
            {
                return information(o);
            });
        } else if(inf == "size") {
            std::cout << vktr.size() << std::endl;
        }
    }
    return 0;
}