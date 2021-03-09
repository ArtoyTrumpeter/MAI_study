import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math as func

fig = plt.figure()
ax = plt.axes(projection='3d')
R = 5

n = 6
print("Center of pyramid(x,y)")
center_x = 0
center_y = 0
z_begin  = 0

z_max = 7

z_fact = 10
k = 1
# print(k)
x_v = list()
y_v = list()
z_v = list()
x_f = list()
y_f = list()
i = 0
while i <= n:
    angle = (2 * func.pi * i) / n;
    x = R * func.cos(angle) + center_x
    y = R * func.sin(angle) + center_y
    x_v.append(x)
    y_v.append(y)
    z_v.append(z_begin)
    x_v.append((R * k * func.cos(angle) + center_x))
    y_v.append((R * k * func.sin(angle) + center_y))
    z_v.append(z_fact)
    x_f.append((R * k * func.cos(angle) + center_x))
    y_f.append((R * k * func.sin(angle) + center_y))
    if(i > 0) :
        x_v.append(x_f[i - 1])
        y_v.append(y_f[i - 1])
        z_v.append(z_fact)
        x_v.append(x_f[i])
        y_v.append(y_f[i])
        z_v.append(z_fact)
    x_v.append(R * func.cos(angle) + center_x)
    y_v.append(R * func.sin(angle) + center_y)
    z_v.append(z_begin)
    i += 1

b = [0,1,2,3]
ax.plot(x_v, y_v, z_v, label='circle1', color='black')
x = np.asarray(x_v)
y = np.asarray(y_v)
z = np.asarray(z_v)
z1=np.expand_dims(z,axis=1)
i = z.shape[0]
p = []
g = []
while i!=0:
    p.append(0)
    g.append(z_fact)
    i = i -  1

ax.plot_surface(x,y, z1, color = 'b')
ax.plot_trisurf(x, y, p, color = 'b')
ax.plot_trisurf(x, y, g, color = 'b')

#ax.add_collection3d(plt.fill_between(x_v, y_v, z_v, color='b', alpha=0.6,label="filled plot", interpolate=True),0, zdir='z')
#ax.add_collection3d(plt.fill_between(x_v, y_v, z_v, color='b', alpha=0.6,label="filled plot"),z_fact, zdir='z')
b = np.array([0,1,2,3])


plt.xlabel('X')
plt.ylabel('Y')
plt.show()
