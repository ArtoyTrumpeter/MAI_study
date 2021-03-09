#include <iostream>
#include "zmq.hpp"
#include <string>
#include <sstream>
#include <exception>
#include <signal.h>
#include "server_functions.h"


int main(int argc, char** argv) { //аргументы - айди и номер порта, к которому нужно подключиться
    int id = std::stoi(argv[1]);
    int parent_port = std::stoi(argv[2]);
    zmq::context_t context(3);
    zmq::socket_t parent_socket(context, ZMQ_REP);
    parent_socket.connect(get_port_name(parent_port));

    int left_pid = 0;
    int right_pid = 0;
    int left_id = 0;
    int right_id = 0;

    
    zmq::socket_t left_socket(context, ZMQ_REQ);
    zmq::socket_t right_socket(context, ZMQ_REQ);
    int linger = 0;
    left_socket.setsockopt(ZMQ_SNDTIMEO, 2000);
    left_socket.setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
    right_socket.setsockopt(ZMQ_SNDTIMEO, 2000);
    right_socket.setsockopt(ZMQ_LINGER, &linger, sizeof(linger));

    int left_port = bind_socket(left_socket);
    int right_port = bind_socket(right_socket);

    while (true) {
        std::string request_string;

        request_string = recieve_message(parent_socket);
        std::istringstream command_stream(request_string);
        std::string command;
        command_stream >> command;

        if (command == "id") {
            std::string parent_string = "Ok:" + std::to_string(id);
            send_message(parent_socket, parent_string);
        } else if (command == "pid") {
            std::string parent_string = "Ok:" + std::to_string(getpid());
            send_message(parent_socket, parent_string);
        } else  if (command == "create") {
            int id_to_create;
            command_stream >> id_to_create;
            // управляюший узел сообщает id нового узла и порт, к которому его надо подключить
            if (id_to_create == id) {
                // если id равен данному, значит узел уже существует, посылаем ответ с ошибкой
                std::string message_string = "Error: Already exists";
                send_message(parent_socket, message_string);
            } else if (id_to_create < id) {
                if (left_pid == 0) {
                    left_pid = fork();
                    if (left_pid == -1) {
                        send_message(parent_socket, "Error: Cannot fork");
                        left_pid = 0;
                    } else if (left_pid == 0) { // в родительском процессе запускается программа
                        create_node(id_to_create,left_port);
                    } else { // в дочернем процессе происходит отправка сообщений
                        left_id = id_to_create;
                        send_message(left_socket, "pid");
                        send_message(parent_socket, recieve_message(left_socket));
                    }
                } else {
                    send_message(left_socket, request_string);
                    send_message(parent_socket, recieve_message(left_socket));
                }
            } else {
                if (right_pid == 0) {
                    right_pid = fork();
                    if (right_pid == -1) {
                        send_message(parent_socket, "Error: Cannot fork");
                        right_pid = 0;
                    } else if (right_pid == 0) {
                        create_node(id_to_create,right_port);
                    } else {
                        right_id = id_to_create;
                        send_message(right_socket, "pid");
                        send_message(parent_socket, recieve_message(right_socket));
                    }
                } else {
                    send_message(right_socket, request_string);
                    send_message(parent_socket, recieve_message(right_socket));
                }
            }

        } else if (command == "remove") {
            int id_to_delete;
            command_stream >> id_to_delete;
            if (id_to_delete < id) {
                if (left_id == 0) {
                    send_message(parent_socket, "Error: Not found");
                } else if (left_id == id_to_delete) {
                    send_message(left_socket, "kill_children");
                    recieve_message(left_socket);
                    kill(left_pid,SIGTERM);
                    kill(left_pid,SIGKILL);
                    left_id = 0;
                    left_pid = 0;
                    send_message(parent_socket, "Ok");

                } else {
                    send_message(left_socket, request_string);
                    send_message(parent_socket, recieve_message(left_socket));
                }
            } else {
                if (right_id == 0) {
                    send_message(parent_socket, "Error: Not found");
                } else if (right_id == id_to_delete) {
                    send_message(right_socket, "kill_children");
                    recieve_message(right_socket);
                    kill(right_pid,SIGTERM);
                    kill(right_pid,SIGKILL);
                    right_id = 0;
                    right_pid = 0;
                    send_message(parent_socket, "Ok");
                } else {
                    send_message(right_socket, request_string);
                    send_message(parent_socket, recieve_message(right_socket));
                }
            }
        } else if (command == "exec") {
            int exec_id;
            command_stream >> exec_id;
            if (exec_id == id) {
                int n;
                command_stream >> n;
                if(n < 0) {
                    n = 0;
                }
                int sum = 0;
                for (int i = 0; i < n; ++i) {
                    int cur_num;
                    command_stream >> cur_num;
                    sum += cur_num;
                }
                std::string recieve_message = "Ok:" + std::to_string(id) + ":" + std::to_string(sum);
                send_message(parent_socket, recieve_message);
                
            } else if (exec_id < id) {
                if (left_pid == 0) {
                    std::string recieve_message = "Error:" + std::to_string(exec_id) + ": Not found";
                    send_message(parent_socket, recieve_message);
                } else {
                    send_message(left_socket, request_string);
                    send_message(parent_socket, recieve_message(left_socket));
                }
            } else {
                if (right_pid == 0) {
                    std::string recieve_message = "Error:" + std::to_string(exec_id) + ": Not found";
                    send_message(parent_socket, recieve_message);
                } else {
                    send_message(right_socket, request_string);
                    send_message(parent_socket, recieve_message(right_socket));
                }
            }

        } else if (command == "ping") {
            std::ostringstream res;
            std::string left_res;
            std::string right_res;
            if (left_pid != 0) {
                send_message(left_socket, "ping");
                left_res = recieve_message(left_socket);
            }
            if (right_pid != 0) {
                send_message(right_socket, "ping");
                right_res = recieve_message(right_socket);
            }
            if (!left_res.empty() && left_res.substr(std::min<int>(left_res.size(),5)) != "Error") {
                res << left_res;
            }


            if (!right_res.empty() && right_res.substr(std::min<int>(right_res.size(),5)) != "Error") {
                res << right_res;
            }
            send_message(parent_socket, res.str());
        } else if (command == "kill_children") {  // УБИТЬ ВСЕХ ДЕТЕЙ
            if (left_pid == 0 && right_pid == 0) {
                send_message(parent_socket, "Ok");
            } else {
                if (left_pid != 0) {
                    send_message(left_socket, "kill_children");
                    recieve_message(left_socket);
                    kill(left_pid,SIGTERM);
                    kill(left_pid,SIGKILL);
                }
                if (right_pid != 0) {
                    send_message(right_socket, "kill_children");
                    recieve_message(right_socket);
                    kill(right_pid,SIGTERM);
                    kill(right_pid,SIGKILL);
                }
                send_message(parent_socket, "Ok");

            }
        }
        if (parent_port == 0) {
            break;
        }
    }

}