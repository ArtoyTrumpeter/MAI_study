#ifndef HISTORY_HPP
#define HISTORY_HPP
#include "figure.hpp"

template <class T>
class History {
public:
    virtual void Cancel() = 0;
};

template <class T>
class CommandRemove : public History<T> {
public:
    CommandRemove(std::shared_ptr<Figure<T>>& at,std::shared_ptr<Document<T>> temp) {
        figure = at;
        doc_ = temp;
    }
    void Cancel() override {
        doc_->SimpleInsert(figure);
    }
private:
std::shared_ptr<Figure<T>> figure;
std::shared_ptr<Document<T>> doc_;
};

template <class T>
class CommandAdd : public History<T> {
public:
    CommandAdd(std::shared_ptr<Document<T>> temp) {
        doc_ = temp;
    }
    void Cancel() override {
        doc_->Remove();
    }
private:
std::shared_ptr<Document<T>> doc_;
};
#endif