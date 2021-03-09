/*создание аллокатора, который использует вектор для занятой памяти и очередь для свободной*/

#ifndef ALLOCATOR_HPP 
#define ALLOCATOR_HPP
#include "square.hpp"
#include "queue.hpp"


template <class T,const size_t sizeblocks>
class TAllocator {
private:
    T* buffer;
    size_t max_size;
    TQueue<T*> freeblocks;
public:
    using value_type = T;
	using pointer = T*;
    using str = TQueue<T*>;
	using const_pointer = const T*;
	using size_type = std::size_t;

    TAllocator() : buffer(nullptr), freeblocks()
    {   
        static_assert(sizeblocks > 0, "Block size can not be lower than 0");
        max_size = sizeblocks;
    };

    size_t Max_Size() {
        return max_size;
    }

    str get_str() {
        return freeblocks;
    }

    T* allocate(size_t n) {//memory allocation
        if(buffer == nullptr) {
            buffer = (T*)malloc(sizeof(T) * sizeblocks);
            ToFull();
            return buffer;
        }
        int i = FreeSpace(n);
        del(i);
        return buffer + i;
    }

    template <class U>
	struct rebind {//struct to create Allocator to different type in our class;
		using other = TAllocator<U, sizeblocks>;
	};

    ~TAllocator() {
        free(buffer);
    }

    void deallocate(T* p,size_t n) {//add blocks in TQueue<T*> freeblocks;
        for(size_t i = 0; i < n;i++) {
            freeblocks.Push(p + i);
        }
        std::cout << "Now free blocks = " << get_str().Size() << std::endl;
    }

    
    template<typename U, typename ...Args>
    void construct(U *p, Args &&...args) {//set in p  Args &&...args(placement new!!!);
        new (p) U(std::forward<Args>(args)...);
    }

    void destroy(T* p) {//return deconstructor T;
        p->~T();
    }
private:
    void ToFull() {
        for(size_t i = 0;i < sizeblocks;i++) {
            freeblocks.Push(buffer + i);
        }
    }

    void del(size_t n) {
        for(size_t i = 0;i < n;i++) {
            freeblocks.Pop();
        }
    }


    size_t FreeSpace(size_t count) {
        size_t i = 0;
        typename TQueue<T*>::Iterator it(freeblocks.Head());
        while((i < count) && (it != freeblocks.end())) {
            it++;
            i++;
        }
        if(i < count) {
            throw std::bad_alloc();
        }
        return i;
    }


};

#endif