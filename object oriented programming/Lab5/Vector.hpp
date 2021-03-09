#include <algorithm>
#include <cassert>
#include <memory>
#include <iostream>

template <typename T>
class TVector { 
    public:
        using value_type = T;
        using iterator = value_type*;
        
        TVector(): 
            already_used_(0), storage_size_(0), storage_(nullptr)
        {
        }

        TVector(int size, const value_type& default_value = value_type()):
            TVector()
        {
            assert(size >= 0);

            if (size == 0) {
                return;
            }

            already_used_ = size;
            storage_size_ = size;
            storage_ = std::make_unique<value_type[]>(size);

            std::fill(storage_.get(), storage_.get() + already_used_, default_value);
        }

        int size() const
        {
            return already_used_;
        }

        bool empty() const
        {
            return size() == 0;
        }

        iterator begin() const
        {
            return storage_.get();
        }
        
        iterator end() const
        {
            if (storage_.get()) {
                return storage_.get() + already_used_;
            }
            return nullptr;
        }
        
        void insert(iterator pos, value_type val) {
            if (already_used_ < storage_size_) {
                std::copy(pos, storage_.get() + already_used_, pos + 1);
                *pos = val;
                ++already_used_;
                return;
            }
            int next_size = 1;
            if (storage_size_) {
                next_size = storage_size_ * 2;
            }
            TVector next(next_size);
            next.already_used_ = already_used_;

            if (storage_.get()) {
                std::copy(storage_.get(), storage_.get() + storage_size_, next.storage_.get());
            }
            next.insert(pos, val);
            Swap(*this, next);
        }
        
        void erase(iterator pos) {
            std::copy(pos + 1, storage_.get() + already_used_, pos);
            --already_used_;
        }
        
        friend void Swap(TVector& lhs, TVector& rhs)
        {
            using std::swap;

            swap(lhs.already_used_, rhs.already_used_);
            swap(lhs.storage_size_, rhs.storage_size_);
            swap(lhs.storage_, rhs.storage_);
        }

        TVector& operator=(TVector other)
        {
            Swap(*this, other);
            return *this;
        }

        TVector(const TVector& other):
            TVector()
        {
            TVector next(other.storage_size_);
            next.already_used_ = other.already_used_;

            if (*(other.storage_) ) {
                std::copy(other.storage_.get(), other.storage_.get() + other.storage_size_,
                        next.storage_.get());
            }

            swap(*this, next);
        }

        ~TVector()
        {
            storage_size_ = 0;
            already_used_ = 0;
        }

        void push_back(const value_type& value)
        {
            if (already_used_ < storage_size_) {
                storage_[already_used_] = value;
                ++already_used_;
                return;
            }
            int next_size = 1;
            if (storage_size_) {
                next_size = storage_size_ * 2;
            }
            TVector next(next_size);
            next.already_used_ = already_used_;

            if (storage_.get()) {
                std::copy(storage_.get(), storage_.get() + storage_size_, next.storage_.get());
            }
            next.push_back(value);
            Swap(*this, next);
        }
        
        value_type& At(int index)
        {
            if (index < 0 || index > already_used_) {
                std::cout << "\nxxxxxx\n";
                throw std::out_of_range("You are doing this wrong!");
            }

            return storage_[index];
        }

        value_type& operator[](int index)
        {
            return At(index);
        }

    private:
        int already_used_;
        int storage_size_;
        std::unique_ptr<value_type[]> storage_;
};