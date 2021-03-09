/*
M80-207Б-18, Тояков Артем Олегович
Спроектировать простейший графический векторный редактор. Требование к функционалу редактора:
    ▪ создание нового документа
    ▪ импорт документа из файла
    ▪ экспорт документа в файл
    ▪ создание графического примитива (согласно варианту задания)
    ▪ удаление графического примитива
    ▪ отображение документа на экране (печать перечня графических объектов и их характеристик)
    ▪ реализовать операцию undo, отменяющую последнее сделанное действие. Должно действовать для операций добавления/удаления фигур.
*/
#include "figure.hpp"
#include "document.hpp"
#include "redactor.hpp"
#include "document.hpp"
    
void Help() {
    std::cout << "add" << std::endl;
    std::cout << "print" << std::endl;
    std::cout << "export" << std::endl;
    std::cout << "import" << std::endl;
    std::cout << "undo"  << std::endl;
    std::cout << "remove" << std::endl;remove
    std::cout << "create(new document)" << std::endl;
    std::cout << "exit" << std::endl;
}

int main() {
    std::cout << "Please,name of document and file" << std::endl;
    std::string name,filename;
    std::cin >> name >> filename;
    Redactor<double> Red(filename,name);
    std::string temp;
    Help();
    while(std::cin >> temp) {
        if(temp == "add") {
           try {
                Red.Insert();    
            } catch (std::logic_error& err) {
				std::cout << err.what() << std::endl;
			}
        } else if(temp == "print") {
            Red.Print();
        } else if(temp == "export") {
           try {
                Red.WriteInFile();    
            } catch (std::logic_error& err) {
				std::cout << err.what() << std::endl;
			}
        } else if(temp == "import") {
            try {
                //импорт из файла
                Red.ReadFromFile();    
            } catch (std::logic_error& err) {
				std::cout << err.what() << std::endl;
			}
        } else if(temp == "undo") {
            try {
                Red.Undo();    
            } catch (std::logic_error& err) {
				std::cout << err.what() << std::endl;
			}
        } else if(temp == "remove") {
            try {
                Red.Remove();    
            } catch (std::logic_error& err) {
				std::cout << err.what() << std::endl;
			}
        } else if(temp == "create") {
            try {
                Red.CreateDoc();    
            } catch (std::logic_error& err) {
				std::cout << err.what() << std::endl;
			}
        } else if(temp == "exit") {
            return 0;
        }
    }
    return 0;
}