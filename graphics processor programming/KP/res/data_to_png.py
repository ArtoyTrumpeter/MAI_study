import os

for i in range(120):
	os.system('python conv.py \\{}.data png\\{}.png'.format(i,i))