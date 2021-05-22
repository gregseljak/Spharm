import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from matplotlib import cm
res = 1000
pi = 3.14159
x = np.linspace(0, 1, res)

def f(x):
    return 1-x+2*x**2
y = f(x)

maxrange = 100
eigenfunctions = np.empty((maxrange, res))
coeffs = np.empty(maxrange)
for n in range(maxrange):
    eigenfunctions[n,:] = (np.sin(n*pi*x))
    coeffs[n] = np.sum(eigenfunctions[n,:] * y/res*2)

f0, ax = plt.subplots(2)
f0.suptitle("Fourier Series Approximation")
ax[0].set_title('True Value')
ax[1].set_title('Approximation')
ax[0].set_xlim(0, 1)
ax[1].set_xlim(0,1)
ax[1].set_ylim(0,50)


# Initializing plots:
# the commas in the next two lines are actually extremely important
itd_plot, = ax[0].plot(x, y)
ifd_plot, = ax[1].plot(x, np.zeros(res))
#plt.show()
#%%
# Creating the acutal frame-to-frame animation
def updateData(frame):

    waveform = coeffs[:frame,None]*eigenfunctions[:frame]

    waveform = np.sum(waveform, axis=0)
    ifd_plot.set_data(x, waveform)
    ax[0].title.set_text([frame])
    return  ifd_plot,

anime = animation.FuncAnimation(f0, updateData, blit=False, frames=maxrange, interval=1000, repeat=False)
f0.tight_layout()
plt.show()