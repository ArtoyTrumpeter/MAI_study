#include "SuffTree.h"
using namespace NSuff;

/*TNode implementation*/

TNode::TNode(const size_t& globalRef, const std::string& origStrRef, size_t lowerBound = 0, size_t upperBound = 0) :
    bounds({lowerBound, upperBound}),
    globalUpper(globalRef),
    strRef(origStrRef)
{

}

TNode* TNode::FindChildByChar(const char ch) {
    TMap::iterator it = children.find(ch);
    if (it == children.end()) {
        return nullptr;
    } else {
        return it->second.get();
    }
}

void TNode::AddLeaf(const size_t lowerBound) {
    children.insert(std::make_pair(
            strRef[lowerBound],
            TUniquePtr( new TNode(globalUpper, strRef, lowerBound) )
    ));
}

TNode* TNode::AddNode(const size_t lowerBound, const size_t upperBound, const char edge) {
    TMap::iterator it = children.find(edge);

    if (it == children.end()) {
        throw std::logic_error("NO_CHILD: when creating node, marked edge not found");
    }
    TUniquePtr foundChild = std::move(it->second);
    foundChild->SetLower(upperBound);

    TUniquePtr newNode = TUniquePtr( new TNode(globalUpper, strRef, lowerBound, upperBound) );
    newNode->children.insert(std::make_pair(
            strRef[upperBound],
            std::move(foundChild)
    ));    
    newNode->AddLeaf(globalUpper - 1);

    TNode* nodeAddress = newNode.get();
    it->second = std::move(newNode);
    return nodeAddress;
}

TNode* TNode::GetSuffixLink() {
    return suffixLink;
}

void TNode::SetSuffixLink(TNode* newNode) {
    suffixLink = newNode;
}

size_t TNode::GetLower() {
    return bounds.lower;
}

void TNode::SetLower(size_t newLower) {
    bounds.lower = newLower;
}

size_t TNode::GetUpper() {
    if (children.empty()) {
        return globalUpper;
    }
    return bounds.upper;
}

size_t TNode::GetLength() {
    return GetUpper() - GetLower();
}

char TNode::GetChar(size_t index) {
    if (index > GetLength()) {
        throw std::out_of_range("NODE_OUT_OF_BOUNDS: Attempting to access character in string beyond node bounds");
    }
    return strRef[GetLower() + index];
}

std::string TNode::GetString() {
    return strRef.substr(GetLower(), GetLength());
}

void TNode::PrintNode(size_t depth) {
    for (size_t i = 0; i < depth; ++i) {
        std::cout << '\t';
    }

    std::cout << GetString() << std::endl;

    for (const std::pair<const char, TUniquePtr>& e : children) {
        e.second->PrintNode(depth + 1);
    }
}

void TNode::FillArray(
        std::vector<size_t>& passedArray,
        size_t rollingLength
) {
    rollingLength += GetLength();

    if (children.empty()) {
        passedArray.push_back(strRef.size() - rollingLength);
        return;
    }

    for (const TPair& it : children) {
        it.second->FillArray(passedArray, rollingLength);
    }
}

/*TSuffTree implementation*/

TSuffTree::TSuffTree() : 
    globalUpper(0), 
    remainder(0),
    activePoint({nullptr, '\0', 0}) 
{
    
}

TSuffTree::TSuffTree(std::string input) :
    TSuffTree()
{
    SetString(input);
    Construct();
}

TSuffTree& TSuffTree::SetString(const std::string input) {
    originalStr = input + '$';
    return *this;
}

TSuffTree& TSuffTree::Construct() {
    globalUpper = 0; 
    remainder   = 0;
    root = TUniquePtr( new TNode(globalUpper, originalStr) );
    activePoint = {root.get(), '\0', 0};
    size_t strSize = originalStr.size();

    for (size_t i = 0; i < strSize; ++i) {
        ++globalUpper;
        ++remainder;

        prevCreated = root.get();
        
        while(remainder) {
            if (activePoint.length == 0) {
                activePoint.edge = originalStr[i];
            }

            TNode* foundChild 
                = activePoint.node->FindChildByChar(activePoint.edge);

            if (foundChild == nullptr) {
                activePoint.node->AddLeaf(i);
                TryLink(activePoint.node);

            } else {
                if (activePoint.length >= foundChild->GetLength()) {
                    WalkDown(i, foundChild);
                    continue;
                }

                if (originalStr[i] == foundChild->GetChar(activePoint.length)) {
                        ++activePoint.length;
                        TryLink(activePoint.node);
                        break;
                }

                size_t nodeLower = foundChild->GetLower();
                TNode* newNode = 
                    activePoint.node->AddNode(nodeLower, nodeLower + activePoint.length, activePoint.edge);
                TryLink(newNode);
            }

            --remainder;
            if (activePoint.node != root.get()) {
                activePoint.node = activePoint.node->GetSuffixLink();
            } else if (activePoint.length > 0) {
                activePoint.edge = originalStr[i - remainder + 1];
                --activePoint.length;
            }
        }
    }

    return *this;
}

void TSuffTree::TryLink(TNode* newNode) {
    if (prevCreated != root.get()) {
        prevCreated->SetSuffixLink(newNode);
    }

    prevCreated = newNode;
}

void TSuffTree::WalkDown(size_t currentIter, TNode* child) {
    TNode* nextChild = child;

    activePoint.length -= nextChild->GetLength();
    activePoint.edge    = originalStr[currentIter - activePoint.length];
    activePoint.node    = nextChild;
}

void TSuffTree::PrintTree() {
    root->PrintNode(0);
}

std::vector<size_t> TSuffTree::CreateSuffArray() {
    std::vector<size_t> array;
    array.reserve(originalStr.size());
    root->FillArray(array, 0);
    return array;
}

/*TSuffArray implementation*/

TSuffArray::TSuffArray(std::vector<size_t> newArray, std::string string) :
    suffArray(newArray)
{
    originalStr = string + '$';
    CalcLcp();
}

std::vector<int> TSuffArray::Find(std::string pattern) {
    std::pair<
        std::vector<size_t>::iterator,
        std::vector<size_t>::iterator
    > ranges (std::make_pair(suffArray.begin(), suffArray.begin()));
    const size_t patSize = pattern.size();
    size_t low  = 0,
           high = originalStr.size() - 1;

    while (low < high) {
        size_t mid = (low + high) / 2;
        if (originalStr.compare(suffArray[mid], patSize, pattern, 0, patSize) >= 0) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    if (originalStr.compare(suffArray[low], patSize, pattern, 0, patSize) != 0) {
        return std::vector<int>();
    }

    ranges.first += low;
    high = originalStr.size() - 1;
    while (low < high) {
        size_t mid = (low + high) / 2;
        if (originalStr.compare(suffArray[mid], patSize, pattern, 0, patSize) > 0) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    if (originalStr.compare(suffArray[high], patSize, pattern, 0, patSize) != 0) {
        --high;
    }
    ranges.second += high + 1;

    std::vector<int> result(ranges.first, ranges.second);
    std::sort(result.begin(), result.end());
    return result;
}

void TSuffArray::PrintArray() {
    std::cout << "{ ";
    for (const int& e : suffArray) {
        std::cout << e << " ";
    }
    std::cout << "}" << std::endl;
}

void TSuffArray::PrintLcp() {
    std::cout << "{ ";
    for (const int& e : lcp) {
        std::cout << e << " ";
    }
    std::cout << "}" << std::endl;
}

void TSuffArray::CalcLcp() {
    size_t size = suffArray.size();

    auto inverseSuff = std::vector<size_t> (size, 0);
    for (size_t i = 0; i < size; ++i) {
        inverseSuff[suffArray[i]] = i;
    }

    lcp = std::vector<size_t> (size - 1, 0);
    size_t prevLcp = 0;
    
    for (size_t i = 0; i < size; ++i) {
        if (inverseSuff[i] == size - 1) {
            prevLcp = 0;
            continue;
        }

        size_t j = suffArray[inverseSuff[i] + 1];

        while (i + prevLcp < size &&
                j + prevLcp < size &&
                originalStr[i + prevLcp] == originalStr[j + prevLcp]
              )
        {
            ++prevLcp;
        }

        lcp[inverseSuff[i]] = prevLcp;

        if (prevLcp > 0) {
            --prevLcp;
        }
    }

}

size_t TSuffArray::GetLCP(size_t lower, size_t upper) {
    return *std::min_element(lcp.begin() + lower, lcp.begin() + upper);
}