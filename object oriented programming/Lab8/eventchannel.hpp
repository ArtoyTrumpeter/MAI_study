#ifndef EVENTCHANNEL_HPP
#define EVENTCHANNEL_HPP
#include "figure.hpp"

// Publish-subscribe
class EventChannel {
    public:
        void subscribe(EventType type,std::shared_ptr<Handler>& new_handler) {
            Handlers.emplace(type,new_handler);
        }

        void doing(EventType type,Event& event) {
            // its is iterator_node, *its = std::pair<const Key, T>;
// its = struct std::__detail::_Node_iterator<std::pair<const EventType, std::shared_ptr<Handler> >, false, false>
            auto its = Handlers.find(type); 
            (*its).second->event(event);
        } 
    private:
    std::unordered_map<EventType,std::shared_ptr<Handler>> Handlers;
};

class EventLoop {
public:
    void addHandler(EventType type,std::shared_ptr<Handler>& new_handler) {
        manager.subscribe(type,new_handler);
    }

    void addEvent(Event& event) {
        std::lock_guard<std::mutex> guard(defend);
        events.push(event);
    }
    void operator()() {
        while(!quit) {
            if(!events.empty()) {
                auto ev = events.front();
                events.pop();
                switch(ev.type) {
                    case EventType::quit:
                        quit = true;
                        ev.ChangeStatus();
                        break;
                    default:
                        manager.doing(ev.type,ev);
                }
            } else {
                // условия сна и без действия!!!
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
    }    
private:
    std::queue<Event> events;
    EventChannel manager;
    bool quit = false;
    std::mutex defend;
};

#endif