//коллекция очереди

#ifndef QUEUE_HPP
#define QUEUE_HPP 
#include "square.hpp"

template <class T> 
class TQueue {
private:
    struct TNode;
    class TIterator;
    using ptr = std::shared_ptr<TNode>;
    size_t size;
    ptr first;
public:
    using value_type = T;
	using size_type = uint64_t;
	using Iterator = TIterator;
    TQueue() : size(0), first(nullptr)
    {}

    void Push(const value_type& new_dt) {
        if(size == 0) {
            first =  std::make_shared<TNode>(new_dt);
        } else {
            ptr temp = first;
            while(temp->next != nullptr) {
                temp = temp->next;
            }
            temp->next = std::make_shared<TNode>(new_dt);
        }
        size++;
    }

    ptr Head() {
        return first;
    }

    size_t Size() {
        return size;
    }
    void Pop() {
        if(size == 0) {
            throw std::logic_error("Queue is empty");
        } 
        first = first->next;
        size--;
    }

    ptr Top() {
        if(size == 0) {
           throw std::logic_error("Queue is empty"); 
        }
        return first;
    }

    Iterator end() {
        return Iterator(nullptr);
    }

    Iterator begin() {
        return Iterator(first);
    }

    Iterator Insert(const ptr& temp, const value_type& new_dt) {
        ptr time = first;
        ptr new_node;
        if(time == temp) {
            new_node = std::make_shared<TNode>(new_dt);
            new_node->next = first;
            first = new_node;
        } else {
            while(time->next != temp) {
                time = time->next;
            }
            new_node = std::make_shared<TNode>(new_dt);
            time->next = new_node;
            new_node->next = temp;
        }
        size++;
        return Iterator(new_node);
    }


    Iterator Find(const int n) {
        if(n == size) {
            return Iterator(nullptr);
        }
        if(n >= size || n < 0) {
            throw std::logic_error("Element on this position was not found");
        }
        int i = 0;
        ptr time = first;
        while(i != n) {
            time = time->next;
            i++;
        }
        return Iterator(time);
    }


    Iterator Erase(const ptr& temp) {
        if(size == 0 || temp == nullptr) {
            throw std::logic_error("Element on this position was not found"); 
        }
        ptr time = first;
        if(time == temp) {
            first = first->next;
        } else {
            while(time->next != temp) {
                time = time->next;
            }
            time->next = temp->next;
        }
        size--;
        return Iterator(time->next);
    } 

private:
    struct TNode {
       value_type data;
       std::shared_ptr<TNode> next;
       TNode(const T& new_data): data(new_data), next(nullptr)
       {}
    };

    class TIterator {
    private:
        ptr it;// ptr = std::shared_ptr<TNode>
        friend class TQueue;
    public:
        using difference_type = uint64_t;
        using value_type = TQueue::value_type;
        TIterator(const ptr& our) : it(our)
        {}

        bool operator ==(const TIterator& other) {
            return (it == other.it);
        }

        bool operator !=(const TIterator& other) {
            return !(it == other.it);
        }

        TIterator& operator++() {//example: ++а
            if(it == nullptr) {
                return (*this);
            }
            it = it->next;
            return *this;
        }

        value_type& operator*() const {
            return (it->data);
        }

        const TIterator operator++(int) {//example : а++
            if(it == nullptr) {
                return (*this);
            }
            const ptr tmp = it;
            it = it->next;
            return TIterator(tmp);
        }  
    };
};

#endif