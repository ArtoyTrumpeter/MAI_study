import socket, threading, time

key = 8194

shutdown = False
join = False

def receving (name, sock):
	while not shutdown:
		try:
			while True:
				data, addr = sock.recvfrom(1024)
				
				print(data.decode("utf-8"))

				time.sleep(0.2)
		except:
			pass
host = socket.gethostbyname(socket.gethostname())
port = 0

server = ("127.0.0.1",9090)

s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.bind((host,port))
s.setblocking(0)

rT = threading.Thread(target = receving, args = ("RecvThread",s))
rT.start()

while shutdown == False:
	try:
		message = input()

		if message != "":
			s.sendto((message).encode("utf-8"),server)
			s.sendto(str(len((message))).encode("utf-8"),server)
		
		time.sleep(0.2)
	except:
		shutdown = True

rT.join()
s.close()