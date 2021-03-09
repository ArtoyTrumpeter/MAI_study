import sys
import matplotlib.pyplot as plt
import numpy as np

# task:
# a > 0
# a is input by user
# ro = a * sin(6 * phi)

print("Please input a > 0:")
a = float(input())
if a <= 0:
    print("EXiting: a must be greater than 0")
    sys.exit(1)

phi = np.linspace(-2 * np.pi, 2 * np.pi, 360)
ro = a * np.sin(12 * phi)

print('Plotting for a =', a)

plt.polar(phi, ro)
plt.show()