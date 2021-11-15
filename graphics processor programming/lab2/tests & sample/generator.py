import random

def gen():
    w = int(input())
    h = int(input())
    with open('test', 'wb') as file :
        file.write(w.to_bytes(4, byteorder='little'))
        file.write(h.to_bytes(4, byteorder='little'))
        for _ in range(w):
            for _ in range(h):
                pixel_data = b''
                pixel_data += random.randint(0, 255).to_bytes(1, byteorder='little')
                pixel_data += random.randint(0, 255).to_bytes(1, byteorder='little')
                pixel_data += random.randint(0, 255).to_bytes(1, byteorder='little')
                pixel_data += (255).to_bytes(1, byteorder='little')
                file.write(pixel_data)
