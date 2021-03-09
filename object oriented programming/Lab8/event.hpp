#ifndef EVENT_HPP
#define EVENT_HPP
#include "figure.hpp"

int randomRange(int low, int high)
{
    return rand() % (high - low - 1) + low;
}

std::string GenFile(int length) {
    srand(time(NULL));
    std::string res = "files/";
    for (int n = 0; n < length; n++)
    {        
        char ch = randomRange('a', 'z');
        res = res + ch;
    }
    res = res + ".txt";
    return res;
}

using data_type = std::vector<std::shared_ptr<Figure<double>>>; 

enum class EventType {
    pif,
    pic,
    quit
};

struct Event {
    public:
    EventType type;
    data_type data;
    int &status;
    Event(EventType beg,data_type our,int &state): type(beg),data(our),status(state)
    {
        if (status != 0) {
		    throw std::logic_error("Status should be zero");
	    }
    }

    void ChangeStatus() {
        this->status = 1;
    }
}; 


class Handler {
public:
    virtual bool event(Event&) = 0;
};


class HandlerPIC : public Handler {
    bool event(Event& ev) override {
        // p is Figure<double>
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        for_each(ev.data.begin(),ev.data.end(),[](auto p) {
            p->Print(std::cout);
        });
        ev.ChangeStatus();
        return true;
    }
};

class HandlerPIF : public Handler {
    bool event(Event& ev) override {
        std::ofstream file;
        file.open(GenFile(8),std::ofstream::app);
        if(!file.is_open()) {
            throw std::logic_error("file don't open");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        for_each(ev.data.begin(),ev.data.end(),[&file](auto p) {
            p->Print(file);
        });
        ev.ChangeStatus();
        return true;
    }
};

#endif