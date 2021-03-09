#ifndef PATRICIA_HPP
#define PATRICIA_HPP

#include <iostream>
#include <fstream>
#include <cstring>

const size_t BUF_SIZE = 257;

class TString {
    char* Str;
    size_t Size;
    using iterator = char *;
public:
    TString() : Str(nullptr), Size(0) {}

    TString(const char* s) {
        Size = strlen(s);
        Str = new char[Size + 1];
        std::copy(s, s + Size, Str);
        Str[Size] = '\0';
    }

    TString(const TString& s) {
        Size = s.Size;
        Str = new char[Size + 1];
        std::copy(s.Str, s.Str + Size, Str);
        Str[Size] = '\0';
    }

    char* String() {
        return Str;
    }

    void Move(char* s) {
		delete[] Str;
		Str = s;
		Size = strlen(s);
	}

    iterator begin() {
        return Str;
    }

    const iterator begin() const {
        return Str;
    }

    iterator end() {
        if (Str)
            return Str + Size;
        return nullptr;
    }

    const iterator end() const {
        if (Str)
            return Str + Size;
        return nullptr;
    }

    size_t StrSize() const {
        return Size;
    }

    char operator[](size_t idx) {
        return Str[idx];
    }

    const char operator[](size_t idx) const {
        return Str[idx];
    }

    friend std::ostream& operator<<(std::ostream& out, const TString& s);
    friend std::istream& operator>>(std::istream& in, const TString& s);

    TString& operator= (const TString& s) {
		delete[] Str;
		Size = s.Size;
		Str = new char[Size + 1];
        std::copy(s.Str, s.Str + Size, Str);
        Str[Size] = '\0';
        return *this;
	}

	TString& operator= (const char* s) {
		delete[] Str;
		Size = strlen(s);
        Str = new char[Size + 1];
        std::copy(s, s + Size, Str);
        Str[Size] = '\0';
		return *this;
	}

    ~TString() {
        delete[] Str;
        Str = nullptr;
        Size = 0;
    }
};

bool operator<(const TString& comp1, const TString& comp2) {
    size_t minSize;
    comp1.StrSize() < comp2.StrSize() ? (minSize = comp1.StrSize()) : (minSize = comp2.StrSize());
    for (size_t i = 0; i < minSize; i++) {
        if (comp1[i] != comp2[i])
            return (comp1[i] < comp2[i]);
    }
    return comp1.StrSize() < comp2.StrSize();
}

bool operator==(const TString& comp1, const TString& comp2) {
    return !(comp1 < comp2) && !(comp2 < comp1);
}

bool operator!=(const TString& comp1, const TString& comp2) {
    return !(comp1 == comp2);
}

std::ostream& operator<<(std::ostream& out, const TString& s) {
    for (auto ch : s)
        out << ch;
    return out;
}

std::istream& operator>>(std::istream& in, TString& s) {
    char buf[BUF_SIZE];
    if (in >> buf)
        s = buf;
    return in;
}

using std::cout;
using std::cin;
using std::endl;


template <typename K, typename V>
class TRBTree {
    enum TColor {R, B};

    struct TNode {
        TColor Color;
        K Key;
        V Value;
        TNode* P;
        TNode* Left;
        TNode* Right;

        TNode(const K& nodeKey, const V& nodeVal) :
        Color(R), Key(nodeKey), Value(nodeVal), P(nullptr), Left(nullptr), Right(nullptr) {}

        TNode() : P(nullptr), Left(nullptr), Right(nullptr) {}
    };
    
    TNode* Root;
    TNode* Nil;
    size_t Size;

    TNode* Uncle(TNode* n) {
        if (n->P == Nil) {
            return Nil;
        }
        if (n->P->P == Nil) {
            return Nil;
        }
        if (n->P->P->Right == n->P) {
            return n->P->P->Left;
        }
        return n->P->P->Right;
    }

public:
    TRBTree() {
        Nil = new TNode;
        Root = Nil;
        Root->Color = B;
    }
    TRBTree(const K& rootKey, const V& rootVal) {
        Nil = new TNode;
        Nil->Color = B;
        Root = new TNode(rootKey, rootVal);
        Root->Color = B;
        Root->P = Nil;
        Root->Left = Nil;
        Root->Right = Nil;
        Size++;
    }

    void Clean(TNode* n) {
        if (n == Nil)
            return ;
        Clean(n->Left);
        Clean(n->Right);
        delete n;
    }

    ~TRBTree() {
        Clean(Root);
        delete Nil;
        Size = 0;
    }

    void Insert(const K& key, const V& val) {
        TNode* n = new TNode(key, val);
        TNode* tmp = Root;
        TNode* p = tmp;
        if (tmp != Nil) {
            while (tmp != Nil) {
                p = tmp;
                if (n->Key < tmp->Key) {
                    tmp = tmp->Left;
                } else {
                    tmp = tmp->Right;
                }
            }
            if (n->Key < p->Key) {
                p->Left = n;
            } else {
                p->Right = n;
            }
        } else {
            Root = n;
        }
        n->Color = R;
        n->P = p;
        n->Left = Nil;
        n->Right = Nil;
        Size++;
        InsertFix(n);
    }

    void InsertFix(TNode* ins) {
        if (ins == Root) {
            ins->Color = B;
            return ;
        }
        if (ins->P->Color == B) {
            return ;
        }
        TNode* unc = Uncle(ins);
        if (unc->Color == R) { //1
            unc->Color = B;
            ins->P->Color = B;
            ins->P->P->Color = R;
            InsertFix(ins->P->P); //max h/2 iterations
        } else {
            if (ins == ins->P->Right && ins->P == ins->P->P->Left) { //2
                LeftRotate(ins->P);
                ins = ins->Left;
            } else if (ins == ins->P->Left && ins->P == ins->P->P->Right) { //3
                RightRotate(ins->P);
                ins = ins->Right;
            }
            if (ins == ins->P->Left) { //4
                RightRotate(ins->P->P);
                ins->P->Color = B;
                ins->P->Right->Color = R;
            } else if (ins == ins->P->Right) { //5
                LeftRotate(ins->P->P);
                ins->P->Color = B;
                ins->P->Left->Color = R;
            }
        }
    }

    void Replace(TNode* n, TNode* rep) {
        if (n == Root)
            Root = rep;
        else
            (n->P->Left == n) ? (n->P->Left = rep) : (n->P->Right = rep);
        rep->P = n->P;
    }

    void Erase(K& key) {
        TNode* n = FindNode(key);
        if (n != Nil) {
            Erase(n);
        }
    }

    void Erase(TNode* n) {
        TNode* del = n;
        TNode* fixNode = Nil;
        TColor delColor = n->Color;
        if (n->Left == Nil) {
            fixNode = n->Right;
            Replace(n, n->Right);
        } else if (n->Right == Nil) {
            fixNode = n->Left;
            Replace(n, n->Left);
        } else {
            TNode* tmp = n->Right;
            while (tmp->Left != Nil) {
                tmp = tmp->Left;
            }
            del = tmp;
            delColor = del->Color;
            fixNode = del->Right;
            if (del->P == n) {
                fixNode->P = del;
            } else {
                Replace(del, del->Right);
                del->Right = n->Right;
                del->Right->P = del;
            }
            Replace(n, del);
            del->Left = n->Left;
            del->Left->P = del;
            del->Color = n->Color; //node replacement (changing Key and Value but Color stays the same)
        }
        if (delColor == B) {
            EraseFix(fixNode);
        }
        delete n;
    }

    void EraseFix(TNode* n) {
        while (n != Root && n->Color == B) {
            if (n->P->Left == n) { //n is Left child
                TNode* bro = n->P->Right;
                if (bro->Color == R) { //1
                    bro->Color = B;
                    bro->P->Color = R;
                    LeftRotate(bro->P);
                    bro = n->P->Right;
                }
                if (bro->Left->Color == B && bro->Right->Color == B) { //2
                    bro->Color = R;
                    n = n->P;
                } else {
                    if (bro->Right->Color == B) { //3
                        bro->Left->Color = B;
                        bro->Color = R;
                        RightRotate(bro);
                        bro = n->P->Right;
                    }
                    bro->Color = bro->P->Color; //4
                    bro->P->Color = B;
                    bro->Right->Color = B;
                    LeftRotate(bro->P);
                    n = Root;
                }
            } else { //n is Right child
                TNode* bro = n->P->Left;
                if (bro->Color == R) {
                    bro->Color = B;
                    bro->P->Color = R;
                    RightRotate(bro->P);
                    bro = n->P->Left;
                }
                if (bro->Left->Color == B && bro->Right->Color == B) {
                    bro->Color = R;
                    n = n->P;
                } else {
                    if (bro->Left->Color == B) {
                        bro->Right->Color = B;
                        bro->Color = R;
                        LeftRotate(bro);
                        bro = n->P->Left;
                    }
                    bro->Color = bro->P->Color;
                    bro->P->Color = B;
                    bro->Left->Color = B;
                    RightRotate(bro->P);
                    n = Root;
                }
            }
        }
        n->Color = B;
    }

    void RightRotate(TNode* pivot) {
        TNode* tmp = pivot->Left;
        pivot->Left = tmp->Right;
        if (tmp->Right != Nil)
            tmp->Right->P = pivot;
        tmp->P = pivot->P;
        if (pivot->P == Nil) {
            Root = tmp;
        } else {
            if (pivot->P->Right == pivot) {
                pivot->P->Right = tmp;
            } else {
                pivot->P->Left = tmp;
            }
        }
        tmp->Right = pivot;
        pivot->P = tmp;
    }

    void LeftRotate(TNode* pivot) {
        TNode* tmp = pivot->Right;
        pivot->Right = tmp->Left;
        if (tmp->Left != Nil)
            tmp->Left->P = pivot;
        tmp->P = pivot->P;
        if (pivot->P == Nil) {
            Root = tmp;
        } else {
            if (pivot->P->Right == pivot) {
                pivot->P->Right = tmp;
            } else {
                pivot->P->Left = tmp;
            }
        }
        tmp->Left = pivot;
        pivot->P = tmp;
    }

    void Print() {
        PrintNode(Root, 0);
        cout << endl << endl;
    }

    void PrintNode(TNode* n, size_t shift) {
        if (n == Nil) {
            return ;
        }
        PrintNode(n->Right, shift + 7);
        for (size_t s = 0; s < shift; s++) {
            cout << " ";
        }
        cout << n->Key << ": " << n->Value << " ";
        n->Color == R ? (cout << "Red" << endl) : (cout << "Black" << endl);
        PrintNode(n->Left, shift + 7);
    }

    TNode* FindNode(K& key) {
        TNode* n = Root;
        while (n->Key != key && n != Nil) {
            if (key < n->Key) {
                n = n->Left;
            } else {
                n = n->Right;
            }
        }
        return n;
    }
};

#endif