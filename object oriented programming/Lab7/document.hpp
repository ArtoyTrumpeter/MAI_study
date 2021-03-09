#ifndef DOCUMENT_HPP
#define DOCUMENT_HPP
#include "figure.hpp"


template <class T>
class Document {
public:
    Document(std::string docname) : buffer(), name(docname),trfactory(), refactory(), sqfactory()
    {}

    std::shared_ptr<Figure<T>> Insert(FigureType type) {
        switch(type) {
            case FigureType::square:
                std::cout << "x1 y1 x2 y2" << std::endl;
                buffer.push_back(sqfactory.CreateFigure(std::cin));
                break;
            case FigureType::triangle:
                std::cout << "x1 y1 x2 y2 x3 y3" << std::endl;
                buffer.push_back(trfactory.CreateFigure(std::cin));
                break;
            case FigureType::rectangle:
                std::cout << "x1 y1 x2 y2" << std::endl;
                buffer.push_back(refactory.CreateFigure(std::cin));
                break;
        }
        return buffer.back();
    }

    std::shared_ptr<Figure<T>> Remove() {
        if(buffer.size() > 0) {
            std::shared_ptr<Figure<T>> temp = buffer.back();
            buffer.pop_back();
            return temp;
        } else {
           throw std::logic_error("Data is empty");
        }
        return nullptr;
    }

    std::shared_ptr<Figure<T>> SimpleInsert(std::shared_ptr<Figure<T>>& at) {
        buffer.push_back(at);
        return buffer.back();
    }

    void Print() {
        std::for_each(buffer.begin(),buffer.end(),[](auto p) {
            p->Print();
        });
    }

    bool WriteInFile(std::string filename) {
        std::ofstream file;
        file.open(filename);
        if(!file.is_open()) {
            throw std::logic_error("file don't open");
        }
        std::for_each(buffer.begin(),buffer.end(),[&file](auto p) {
            p->PrintInFile(file);
        });
        file.close();
        return true;
    }

    void Clear() {
        buffer.clear();
    }
    size_t size() {
        return buffer.size();
    }

    std::vector<std::shared_ptr<Figure<T>>> GetData() {
        return buffer;
    } 

    bool ReadFromFile(std::string filename) {
        std::ifstream file;
        file.open(filename);
        if(!file.is_open()) {
            throw std::logic_error("file don't open");
        }
        std::string type;
        while(!file.eof()) {
            type = "";
            file >> type;
            if(type == "r") {
                buffer.push_back(refactory.CreateFigure(file));
            } else if(type == "t") {
                buffer.push_back(trfactory.CreateFigure(file));
            } else if(type == "s") {
                buffer.push_back(sqfactory.CreateFigure(file));
            }
        }
        file.close();
        return true;
    }
private:
    std::vector<std::shared_ptr<Figure<T>>> buffer;
    std::string name;
    TriangleFactory<T> trfactory;
    RectangleFactory<T> refactory;
    SquareFactory<T> sqfactory;
};

#endif