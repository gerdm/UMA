from matplotlib import animation, pyplot as plt
import numpy as np
from numpy import sin, cos, pi
from numpy.random import binomial

nperiods = 100
fig = plt.figure()

ax = fig.add_subplot(111)

mapf = lambda x, t: cos(pi * t) if x == 0 else sin(pi * t)
Mapf = np.vectorize(mapf)


X = binomial(1, 0.5, nperiods)
T = np.arange(nperiods)
Xt = Mapf(X, T)

def plot_steps(i):
    plt.plot(T[:i], Xt[:i])

animate = animation.FuncAnimation(fig, plot_steps, frames=nperiods)
plt.show()
