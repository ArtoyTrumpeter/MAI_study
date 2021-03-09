#ifndef REDACTOR_HPP
#define REDACTOR_HPP
#include "figure.hpp"
#include "document.hpp"
#include "history.hpp"


template <class T>
class Redactor {
private:
    std::shared_ptr<Document<T>> doc;
    std::deque<std::shared_ptr<History<T>>> history;
    std::string filename;
public:
    Redactor(std::string fname, std::string docname): doc(std::make_shared<Document<T>>(docname)),history(), filename("files/" + fname + ".txt")
    {}

    void CreateDoc() {
        std::string new_name;
        std::cout << "Please, name of document?" << std::endl;
        std::cin >> new_name;
        std::cout << "Are you want to work with other file?(Y or N)" <<std::endl;
        std::string temp;
        std::cin >> temp;
        if(temp == "Y") {
            std::string new_file_name;
            std::cin >> new_file_name;
            new_file_name = "files/" + new_file_name + ".txt";
            filename = new_file_name;
        } else {
            std::cout << "File not change" << std::endl;
        }
        std::shared_ptr<Document<T>> new_doc(std::make_shared<Document<T>>(new_name));
        doc = new_doc;
        history.clear();
    }

    void WriteInFile(){
        doc->WriteInFile(filename);
    }

    void ReadFromFile() {
        history.clear();
        doc->Clear();
        doc->ReadFromFile(filename);
        for(size_t i = 0;i < doc->size();i++) {
            history.push_back(std::shared_ptr<History<T>>(new CommandAdd<T>(doc))); 
        }
    }

    void Insert() {
        std::cout << "Square = s, Triangle = t, Rectangle = r" << std::endl;
        std::string type;
        std::cin >> type;
        if(type == "r") {
            doc->Insert(FigureType::rectangle);
        } else if(type == "t") {
            doc->Insert(FigureType::triangle);
        } else if(type == "s") {
            doc->Insert(FigureType::square);
        } else {
            throw std::logic_error("Unknown key");
        }
        history.push_back(std::shared_ptr<History<T>>(new CommandAdd<T>(doc)));
    }

    void Print() {
        doc->Print();
    }

    void Remove() {
        std::shared_ptr<Figure<T>> temp = doc->Remove();
        history.push_back(std::shared_ptr<History<T>>(new CommandRemove<T>(temp,doc)));
    }

    void Undo() {
       if (history.empty()) {
        throw std::logic_error("History is empty");
    }
    std::shared_ptr<History<T>> lastCommand = history.back();
    lastCommand->Cancel();
    history.pop_back();
    }
};

#endif