//коллекция вектора

#ifndef VECTOR_HPP
#define VECTOR_HPP
#include "square.hpp" 
#include "allocator.hpp"


template <class T,class alloc>
class TVector {
private:
    class TIterator;
public:
    using value_type = T;
    using Iterator = TIterator;
    using ptr = value_type*;
    TVector() : length(0),capacity(0),storage(nullptr)
    {}

    TVector(long long size_v) : length(0), capacity(size_v) 
    {
        assert((long long)Allocator.Max_Size() > size_v);
        storage = Allocator.allocate(size_v);
    }

    TVector(const TVector& other)//constructor with parameter = vector;
    {
        length = other.length;
        capacity = other.capacity;
        storage = Allocator.allocate(capacity);
        if (other.storage) {
            for(int i = 0; i < length; i++) {
                Allocator.allocate(1);
                Allocator.construct(storage + i,other.storage[i]);
            }
        }
    }


    long long size() const {
        return length;
    }

    void PushBack(const value_type& dt) {
        if(length < capacity) {
            Allocator.allocate(1);
            Allocator.construct(storage + length,std::move(dt));
            length++;
            return;
        }
        if(length >= (long long)Allocator.Max_Size()) {//checking Allocator's block size
            throw std::logic_error("Free memory is zero");
        }
        capacity = capacity * 2;
        if(capacity > (long long)Allocator.Max_Size()) {//checking Allocator's block size
            capacity = (long long)Allocator.Max_Size();
        }
        Allocator.allocate(1);
        Allocator.construct(storage + length,std::move(dt));
        length++;
    }

    void PopBack() {
        if(length == 0) {
            throw std::logic_error("Vector is empty");
        }
        Allocator.destroy(storage + length - 1);
        Allocator.deallocate(storage + length - 1,1);
        length--;
    }

    Iterator begin() {
        return storage;
    }

    Iterator end() {
        if(storage) {
            return storage + length;
        }
        return nullptr;
    }

    Iterator Erase(long long position) {
        if(position < 0 || position >= length) {
            throw std::logic_error("Element on this position not found");
        }
        Allocator.destroy(storage + position);
        for(long long i = position; i < length - 1;i++) {
            storage[i] = std::move(storage[i + 1]);
        }
        length--;
        Allocator.destroy(storage + length);
        Allocator.deallocate(storage + length,1);
        return Iterator(storage + position);
    }


    TVector& operator=(const TVector& other)//copying information on this;
    {   
        for(int i = 0;i < length;i++) {
            Allocator.destroy(storage + i);
        }
        Allocator.deallocate(storage,length);
        if(capacity < other.capacity) {
           storage = Allocator.allocate(other.length); 
        }
        length = other.length;
        capacity = other.capacity;
        if (other.storage) {
            for(int i = 0; i < length; i++) {
                Allocator.allocate(1);
                Allocator.construct(storage + i,other.storage[i]);
            }
        }
        return *this;
    }

    value_type& at(long long index)//std::at;
    {
        if (index < 0 || index > length) {
            throw std::out_of_range("You are doing this wrong!");
        }

        return *(storage + index);
    }

    value_type& operator[](long long index)
    {
        return at(index);
    }
    
    ~TVector() {
        for(int i = 0;i < length;i++) {
            Allocator.destroy(storage + i);
        }
        Allocator.deallocate(storage,length);
    }
    
private:
    long long length;
    long long capacity;
    alloc Allocator;
    ptr storage;
    class TIterator {
        ptr it;
        friend class TVector;
    public:
        using difference_type = uint64_t;
        using value_type = TVector::value_type;
        TIterator(const ptr& our) : it(our)
        {}

        bool operator ==(const TIterator& other) {
            return (it == other.it);
        }

        bool operator !=(const TIterator& other) {
            return !(it == other.it);
        }

        TIterator& operator++() {//для префиксной(++а)
            if(it == nullptr) {
                return (*this);
            }
            it = it + 1;
            return *this;
        }

        value_type& operator*() const {
            return (*it);
        }

        const TIterator operator++(int) {//для постфиксной(а++)
            if(it == nullptr) {
                return (*this);
            }
            const ptr tmp = it;
            it = it + 1;
            return TIterator(tmp);
        }  
    };
};
#endif