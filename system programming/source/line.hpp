#ifndef LINE_HPP
#define LINE_HPP
/*
left -> true = [, false = (
*/ 
// infinity = std::numeric_limits;

//

template < class T>
class Line {
public:
    friend class ContextFree;


    Line(std::string& str) {
        auto it = str.begin();
        this->left = false;
        this->a = 0;
        this->b = 0;
        this->right = false;
        CheckBracket(it);
        T lSign;
        CheckSign(it, lSign);
        if(*it == '&') {
            this->a = std::numeric_limits<T>::min();
            it++;
        } else {
            GetDigit(it,a);
            this->a = lSign * a;
        }
        it++;
        T rSign;
        CheckSign(it, rSign);
        if(*it == '&') {
            this->b = std::numeric_limits<T>::max();
            it++;
        } else {
            GetDigit(it,b);
            this->b = rSign * b;
        }
        CheckBracket(it);
    }


    Line(bool nleft, bool nlinfinity, T& na, T& nb, bool nright, bool nrinfinity) :
        left(nleft),  a(na), b(nb), right(nright)
    {}


    void Print() {
        std::cout << left << "," << a << "," << b << "," << right << std::endl;
    }


    bool Merge(std::shared_ptr<Line<T>>& other) {
        int st = 0;
        if(this->a < other->a && this->b > other->b) /* a oa ob b */ {
            //std::cout << "1" << std::endl;
            return true;
        } else if(other->a <= this->a && other->b >= this->b) /* oa <-a b-> ob */ {
            if(other->a == this->a) {
                if(other->left == true) {
                    this->left = true;
                }
            } else if(other->a < this->a) {
                this->a = other->a;
                this->left = other->left;
            }
            if(other->b == this->b) {
                if(other->right == true) {
                    this->right = true;
                }
            } else if(other->b > this->b) {
                this->b = other->b;
                this->right = other->b;
            }
            //std::cout << "2" << std::endl;
            st++;
        } else if(((this->a == other->b) && (this->left == true || other->right == true)) ||
        ((other->a < this->a) && (this->b > other->b && this->a < other->b))) /* oa a,ob-> b */ {
            this->a = other->a;
            this->left = other->left;
            st++;
            //std::cout << "3" << std::endl;
        } else if(((other->a == this->b) && (this->right == true || other->left == true)) ||
        ((other->a > this->a && this->b > other->a) && (other->b > this->b))) /* a oa-> b ob */ {
            this->b = other->b;
            this->right = other->right;
            st++;
            //std::cout << "4" << std::endl;
        } 
        return st > 0;
    }
private:
    bool left;
    T a;
    T b;
    bool right;


    void CheckBracket(std::string::iterator& it) {
        if(*it == '[') {
            left = true;
        }
        if(*it == ']') {
            right = true;
        }
        it++;
    }


    void CheckSign(std::string::iterator& it,T& sign) {
        sign = 1;
        if(*it == '+') {
            it++;
        }
        if(*it == '-') {
            sign = -1;
            it++;
        }
    }


    void GetDigit(std::string::iterator& it,T& a) {
        while(std::isdigit(*it)) {
            a = a * 10 + (*it - '0');
            it++;
        }
    }
};

#endif