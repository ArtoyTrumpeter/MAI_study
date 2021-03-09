import socket, time

host = "127.0.0.1"
port = 9090

clients = []

s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.bind((host,port)) #запустил сервер по ip и порту

quit = False
print("[ Server Started ]")
i = -1
j = 0
while not quit:
	try:
		data, addr = s.recvfrom(1024) #получил msg и unique num

		if addr not in clients: #если это новый клиент то добавляю адрес этого клиента
			clients.append(addr)
		if(j!= 0):
			if(i == 0):
				print(data.decode("utf-8"))
		j = j + 1
		for client in clients: 
			if addr != client:
				if(i == 1):
					res = str(reciv) + " " + str(data.decode("utf-8"))
					s.sendto(res.encode("utf-8"),client)
			elif (i == 0):
				s.sendto(("I got the string").encode("utf-8"), client)
				reciv = len(data.decode("utf-8"))
				i = i + 1
			else :
				i = 0

			
	except:	
		print("\n[ Server Stopped ]")
		quit = True
		
s.close()