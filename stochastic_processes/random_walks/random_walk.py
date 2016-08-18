import numpy as np
from matplotlib import animation, style, pyplot as plt
from numpy.random import binomial
style.use("ggplot")

class Random_Walk(object):
    def __init__(self, p, nperiods):
        self.p = p
        self.nperiods = nperiods
        self.steps = self.simulate()

    def simulate(self):
        vals = binomial(1, self.p, self.nperiods)
        vals = np.piecewise(vals, [vals==1, vals==0], [1, -1])
        print(vals)
        return np.cumsum(vals)

rwalk = Random_Walk(0.5, 1000)
fig = plt.figure()
ax = fig.add_subplot(111)

def iterate_simul(i):
    ax.clear()
    ax.set_xlim([-1, rwalk.nperiods+1])
    ax.set_ylim([min(rwalk.steps), max(rwalk.steps)])
    step = rwalk.steps[:i]
    ax.plot(range(i), step)

anim = animation.FuncAnimation(fig, iterate_simul, frames=rwalk.nperiods, blit=False)
anim.save("rwalk.gif", fps=60,  writer="imagemagick")
plt.show()
