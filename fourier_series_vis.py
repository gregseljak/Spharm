import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from matplotlib import cm
res = 1000
pi = 3.14159
x = np.linspace(0, 1, res)

###################################################
###                    README                   ###
### ------------------------------------------- ###
### Companion piece for harmonics_vis.py, to    ###
### motivate the use of infinite sums of        ###
### eigenfunctions in approaching a function on ###
### a finite interval.                          ###
###                                             ###
### change the definition of f(x) to change the ###
### arbitrary function you wish to model.       ###
### The animation shows the sum of the first n  ###
### fourier series terms.                       ###
###################################################

def f(x):
    """ 
    the function to be approximated:
        it can be any (square-integrable) function!
        Discontinuities are not a problem.
    """
    f_base = np.exp(x)
    f_base[400:650] = 2
    return f_base
    #return 1-0.8*np.cos(x**2)
    
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
axd = np.max(y)-np.min(y)
plot_margin = 0.15
ax[0].set_ylim(np.min(y)-plot_margin*axd, np.max(y)+plot_margin*axd)
ax[1].set_ylim(np.min(y)-plot_margin*axd, np.max(y)+plot_margin*axd)


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