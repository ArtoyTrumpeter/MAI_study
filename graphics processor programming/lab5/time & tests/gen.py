import random

def gen():
    n = int(input())
    with open('test', 'wb') as file :
        file.write(n.to_bytes(4, byteorder='little'))
        for _ in range(n):
            data = b''
            data += random.randint(0, 1000).to_bytes(4, byteorder='little')
            file.write(data)

gen()