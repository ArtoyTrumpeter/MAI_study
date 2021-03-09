#ifndef SUFF_TREE_H
#define SUFF_TREE_H

#include <map>
#include <vector>
#include <algorithm>
#include <memory>
#include <string>
#include <iostream>
#include <exception>

namespace NSuff {
    
    class TNode {
    public:
        TNode(const size_t&, const std::string&, size_t, size_t);
        ~TNode() { };

        TNode* FindChildByChar(const char);
        void   AddLeaf(const size_t);
        TNode* AddNode(const size_t, const size_t, const char);

        TNode* GetSuffixLink();
        void   SetSuffixLink(TNode*);

        size_t GetLower();
        void   SetLower(size_t);

        size_t GetUpper();

        size_t      GetLength();
        char        GetChar(size_t);
        std::string GetString();


        void PrintNode(size_t);

        void FillArray(std::vector<size_t>&, size_t);

    private:
        typedef std::unique_ptr<TNode>      TUniquePtr;
        typedef std::map<char, TUniquePtr>  TMap;
        typedef std::pair<const char, TUniquePtr> TPair;

        TMap   children;
        TNode* suffixLink;
        struct {
            size_t lower;
            size_t upper;
        } bounds;
        const size_t&      globalUpper;
        const std::string& strRef;

    };

    class TSuffTree {
    public:
        TSuffTree();
        TSuffTree(std::string);

        TSuffTree& SetString(const std::string);
        TSuffTree& Construct();

        void       PrintTree();

        std::vector<size_t> CreateSuffArray();

    private:
        typedef std::unique_ptr<TNode> TUniquePtr;
     
        TNode*      prevCreated;
        TUniquePtr  root;
        std::string originalStr;
        size_t      globalUpper;
        size_t      remainder;

        struct {
            TNode* node;
            char   edge;
            size_t length;
        } activePoint;

        void IncrementGlobal();
        void TryLink(TNode*);
        void WalkDown(size_t, TNode*);
    };

    class TSuffArray {
    public:
        TSuffArray(std::vector<size_t>, std::string);
        
        std::vector<int> Find(std::string);

        void PrintArray();
        void PrintLcp();

    private:
        typedef std::vector<size_t> TArray;

        TArray suffArray;
        TArray lcp;
        std::string originalStr;
        
        void   CalcLcp();
        size_t GetLCP(size_t min, size_t max);
    };
}

#endif