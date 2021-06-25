import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from matplotlib import cm
res = 1000
pi = 3.14159
x = np.linspace(-1, 1, res)

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
    #return f_base

    return x
    
y = f(x)

maxrange = 45
eigenfunctions = np.empty((maxrange, res), dtype='complex')
coeffs = np.empty(maxrange, dtype='complex')
for n in range(maxrange):
    eigenfunctions[n,:] = (np.cos(n*np.pi*x) - 1j*np.sin(n*np.pi*x))
    coeffs[n] = 2*np.sum(eigenfunctions[n,:] * y/res)

f0, ax = plt.subplots(2)
f0.suptitle("Fourier Series Approximation")
ax[0].set_title('Included Terms')
ax[1].set_title('Approximation')
ax[0].set_xlim(-0.5, len(coeffs))
plot_margin = 0.15
ax[1].plot(x,np.real(y), color="gray")
ax[1].plot(x,np.imag(y), linestyle="--", color="gray")


# Initializing plots:
freqrange = np.arange(0,len(coeffs),1) -1
ax[0].vlines(freqrange, np.zeros(len(coeffs)), np.real(coeffs), color="gray")
ax[0].vlines(freqrange+0.1, np.zeros(len(coeffs)), np.imag(coeffs), color="gray", linestyle="--")
real_plot, = ax[1].plot(x, np.zeros(res), color="C0")
imag_plot, = ax[1].plot(x, np.zeros(res), color="C1")
#plt.show()
#%%
# Creating the acutal frame-to-frame animation
def updateData(frame):

    waveform = coeffs[:frame,None]*eigenfunctions[:frame]
    waveform = np.flip(np.sum(waveform, axis=0))
    ax[0].vlines(freqrange[frame], 0, np.real(coeffs[frame]), color="C0")
    ax[0].vlines(freqrange[frame]+0.1, 0, np.imag(coeffs[frame]), color="C1")
    real_plot.set_data(x, np.real(waveform))
    imag_plot.set_data(x, np.imag(waveform))
    ax[0].title.set_text([frame-1])
    #return  ifd_plot,

anime = animation.FuncAnimation(f0, updateData, blit=False, frames=maxrange, interval=500, repeat=False)
f0.tight_layout()
plt.show()