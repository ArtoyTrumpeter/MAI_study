/*
M80-207Б-18 Тояков Артем
Создать приложение, которое будет считывать из стандартного ввода данные фигур,
согласно варианту задания, выводить их характеристики на экран и записывать в файл.
Фигуры могут задаваться как своими вершинами,
так и другими характеристиками (например, координата центра, количество точек и радиус).
Вариант 25: треугольник, квадрат, прямоугольник.
*/

#include "figure.hpp"
#include "event.hpp"
#include "eventchannel.hpp"
#define Error_size 3

void help() {
    std::cout << "add -> add figure" << std::endl;
    std::cout << "quit -> end working" << std::endl;
}

int main(int argc, char* argv[]) {
    if(argc != 2) {
        std::cout << "Use ./main size" << std::endl;
        exit(Error_size);
    }
    int buf_size = std::atoi(argv[1]);
    if(buf_size <= 0) {
        std::cout << "Error, buf-" << std::endl;
        exit(Error_size);
    }
    help();
    std::vector<std::shared_ptr<Figure<double>>> figures;
    EventLoop eventLoop;
    std::thread workerThread(std::ref(eventLoop));
    /*
    1-й вариант
    std::shared_ptr<Handler> handlerPIC = std::make_shared<HandlerPIC>();
    std::shared_ptr<Handler> handlerPIF = std::make_shared<HandlerPIF>();
    2-й вариант 
    std::shared_ptr<Handler> handlerPIF(std::make_shared<HandlerPIF>());
    std::shared_ptr<Handler> handlerPIC(std::make_shared<HandlerPIC>());
    2-й вариант практически идентичен 3-ему
    */
    // 3-й вариант
    std::shared_ptr<Handler> handlerPIF(new HandlerPIF());
    std::shared_ptr<Handler> handlerPIC(new HandlerPIC());
    eventLoop.addHandler(EventType::pic,handlerPIC);
    eventLoop.addHandler(EventType::pif,handlerPIF);
    std::string term;
    while(std::cin >> term) {
        if (term == "quit") {
            int a = 0;
            Event ev(EventType::quit,figures,a);
            eventLoop.addEvent(ev);
            workerThread.join();
            return 0;
        } else if (term == "add") {
            std::string temp;
            std::cout << "t - triangle, s - square, r - rectangle" << std::endl;
            std::cin >> temp;
            if(temp == "r") {
                double x1,y1,x2,y2;
                std::cout << "x1 y1 x2 y2" << std::endl;
                std::cin >> x1 >> y1 >> x2 >> y2;
                figures.push_back(std::make_shared<Rectangle<double>>(x1,y1,x2,y2));
            } else if(temp == "t") {
                double x1,y1,x2,y2,x3,y3;
                std::cout << "x1 y1 x2 y2 x3 y3" << std::endl;
                std::cin >> x1 >> y1 >> x2 >> y2 >> x3 >> y3;
                try {
                    figures.push_back(std::make_shared<Triangle<double>>(x1,y1,x2,y2,x3,y3));
                } catch (std::logic_error& err) {
				    std::cout << err.what() << std::endl;
			    }
            } else if(temp == "s") {
                double x1,y1,x2,y2;
                std::cout << "x1 y1 x2 y2" << std::endl;
                std::cin >> x1 >> y1 >> x2 >> y2;
                try {
                    figures.push_back(std::make_shared<Square<double>>(x1,y1,x2,y2));
                } catch (std::logic_error& err) {
				    std::cout << err.what() << std::endl;
			    }
            }
        }
        if (figures.size() == (size_t)buf_size) {
            try {
                int f_st = 0,s_st = 0;
                Event ev1(EventType::pic,figures,f_st);
                Event ev2(EventType::pif,figures,f_st);
                eventLoop.addEvent(ev1);
                eventLoop.addEvent(ev2);
                std::cout << "wait..." << std::endl;
                while(f_st != 1 && s_st != 1) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                figures.resize(0);
            } catch (std::logic_error& err) {
				    std::cout << err.what() << std::endl;
			}
        }
    
    }
    workerThread.join();
    return 0;
}