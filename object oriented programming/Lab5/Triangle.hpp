#include <iostream>
#include <iomanip>
#include <cmath>

template <class A, class B> 
class PairWIO : public std::pair<A, B> {
    public:
        PairWIO() : std::pair<A, B>() {
            this->first = 0;
            this->second = 0; 
        }
        PairWIO(A firstI, B secondI) : std::pair<A, B>() {
            this->first = firstI;
            this->second = secondI; 
        }
        friend std::ostream& operator<< (std::ostream &out, const PairWIO<A,B> &point) {
            out << std::fixed << std::setprecision(2) << '(' << point.first << ", " << point.second << ')';
            return out;
        }
        friend std::istream& operator>> (std::istream &in, PairWIO<A,B> &point) {
            in >> point.first;
            in >> point.second;
            return in;
        } 
};

template <class T>
double distance(PairWIO<T,T> one, PairWIO<T,T> two) {
    return sqrt((one.first - two.first) * (one.first - two.first) + (one.second - two.second) * (one.second - two.second));
}

template <class T> 
class Figure {
    public:
        using Point = PairWIO<int,int>;
        Point points[3];
};

template <class T> 
class Triangle : public Figure<T> {
    public:
        using Point = PairWIO<int,int>;
        Triangle(Point one, Point two, Point three) {
            Figure<T>::points[0] = one;
            Figure<T>::points[1] = two;
            Figure<T>::points[2] = three;
            double x1, x2, x3;
            x1 = distance(one, two);
            x2 = distance(two, three);
            x3 = distance(three, two);
            if (((x1 + x2) <= x3) || ((x2 + x3) <= x1) || (x1 + x3 <= x2)) {
                throw std::logic_error("This is not triangle");
            }
        }
};

template <typename T>
constexpr bool IsTuple = false;
template<typename ... types>
constexpr bool IsTuple<std::tuple<types...>>   = true;

template <class T, 
typename  std::enable_if<std::tuple_size<T>::value == 3>::type* = nullptr> 
void printCoor(T figure) {
    std::cout << "1st = " << std::get<0>(figure) << "\n2nd = " << std::get<1>(figure) << "\n3rd = " << std::get<2>(figure) << '\n';
}

template <class T, 
typename  std::enable_if<!(IsTuple<T>)>::type* = nullptr> 
void printCoor(T figure) {
    std::cout << "1st = " << figure.points[0] << "\n2nd = " << figure.points[1] << "\n3rd = " << figure.points[2] << '\n';
}

template <class T, 
typename  std::enable_if<std::tuple_size<T>::value == 4>::type* = nullptr>
auto centr(T figure) {
    PairWIO<double,double> out;
   
    out.first += std::get<0>(figure).first;
    out.second += std::get<0>(figure).second;
    out.first += std::get<1>(figure).first;
    out.second += std::get<1>(figure).second;
    out.first += std::get<2>(figure).first;
    out.second += std::get<2>(figure).second;
    
    out.first /= 3;
    out.second /= 3;
    return out;
}

template <class T, 
typename  std::enable_if<!(IsTuple<T>)>::type* = nullptr>
auto centr(T figure) {
    PairWIO<double,double> out;
    for (int i = 0; i < 3; i++) {
        out.first += figure.points[i].first;
        out.second += figure.points[i].second;
    }
    out.first /= 3;
    out.second /= 3;
    return out;
}

template <class T>
double geron(PairWIO<T,T> one, PairWIO<T,T> two, PairWIO<T,T> three) {
    double a = distance(one, two);
    double b = distance(two, three);
    double c = distance(one, three);
    double p = (a + b + c) / 2;
    return sqrt(p * (p - a) * (p - b) * (p - c));
}

template <class T, 
typename  std::enable_if<!(IsTuple<T>)>::type* = nullptr>
double area(T figure) { 
    return geron(figure.points[0], figure.points[1], figure.points[2]);
}

template <class T, 
typename  std::enable_if<std::tuple_size<T>::value == 3>::type* = nullptr>
double area(T figure) {
    return geron(std::get<0>(figure), std::get<1>(figure), std::get<2>(figure));
}