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
        return np.cumsum(vals)

nsteps = 500
rwalk1 = Random_Walk(0.5, nsteps)
rwalk2 = Random_Walk(0.7, nsteps)
rwalk3 = Random_Walk(0.3, nsteps)
all_steps = np.hstack([rwalk1.steps, rwalk2.steps, rwalk3.steps])
print(all_steps)
ymin = all_steps.min()
ymax = all_steps.max()

fig = plt.figure()
ax = fig.add_subplot(111)

def iterate_simul(i):
    ax.clear()
    ax.set_xlim([-1, rwalk1.nperiods+1])
    ax.set_ylim([ymin, ymax])
    i+=1
    step1 = rwalk1.steps[:i]
    step2 = rwalk2.steps[:i]
    step3 = rwalk3.steps[:i]
    ax.plot(range(i), step1, label=r"p=0.5")
    ax.plot(range(i), step2, label=r"p=0.7")
    ax.plot(range(i), step3, label=r"p=0.3")

    plt.legend()

anim = animation.FuncAnimation(fig, iterate_simul, frames=nsteps, blit=False)
anim.save("rwalk.gif", fps=60,  writer="imagemagick")
plt.show()
