
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.special import sph_harm
import matplotlib.animation as animation
from matplotlib import cm
#################################################
###     ------     USER README    ------      ###
###                                           ###
### The field can be manipulated through the  ###
### choice of coefficients in this list       ###
### e.g. C(l=3, m=-1) = coeffs[3][2]          ###
###                                           ###
########################## Greg Seljak, 2021  ###
#################################################
coeffs = [[0],[0,1,1],[0,0,0,0,2],[0,0,0,0,0,0,0]]

#%%
##################################################
# You don't need to rerun the program each time! #
# Run this file is in interactive console:       #
#   $python3 -i harmonics_vis.py                 #
#                                                #
# Now you can manipulate and re-run the visuals  #
# without having to recompute the fields:        #
#                                                #
# >>> FG.coeffs = [[0.4], [0+2j,0.1,1]]          #
# >>> FG.show_frame()                            #
# >>> FG.show_movie()                            #
##################################################
#
# N.B. To comply with scipy's convention, I follow 
# this coordinate system :
# (theta, phi)                     # (0,0)
#  N,S,E,W :     (pi/2, -pi) #         # (pi/2, pi)
#                         (pi, 0) #
#
################################################


def main():
    FG = FieldGenerator()
    FG.show_movie()
    return FG

class FieldGenerator(object):

    def __init__(self, imagepath="./worldmap.png"):
        self.world_img = mpimg.imread(imagepath)
        self.y_ind, self.x_ind = self.world_img.shape[0], self.world_img.shape[1]
        self.coeffs = coeffs
        self.globalmax = 0
        self.globalmin = 0
        self.generate_eigenfields()

    def depth_movie(self, nb_frames):
        self.radii = np.linspace(0.5, 1.5, nb_frames)
        film_reel = np.zeros((nb_frames, self.y_ind, self.x_ind), dtype=complex)
        for i in range(nb_frames):
            film_reel[i,:,:] = self.generate_frame(self.radii[i])
        self.film_reel = film_reel    

    def ct_sp(self, x,y):
        # Certainly the most difficult part of the project!
        x_hat = 2*x/self.x_ind - 1
        y_hat = -2*y/self.y_ind + 1
        
        theta = np.arccos(y_hat)
        if (np.abs(y_hat) == 1) and (x == int(self.x_ind/2)):
            phi = 0 
        elif ((x_hat**2 + y_hat**2) <= 1):
            phi = 2*np.arcsin(x_hat)/(np.abs(np.sin(theta)))
        else:
            phi = np.nan
        return(theta, phi)

    def show_field(self, field):
        realfield = np.real(field)
        fig = plt.imshow(self.world_img)
        plt.imshow(realfield, alpha=0.8, cmap='seismic', vmin=np.min(realfield), vmax=np.max(realfield), interpolation="none")
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.show()

    def show_point(self, pointx, pointy):
        fig = plt.imshow(self.world_img)
        plt.scatter(pointx, pointy, s=5)
        plt.show()

    def generate_eigenfields(self):
        """
        Generate a ragged list of complex-valued fields over the image.
        Computationally intensive, but it only has to be done once.
        """
        eigenfields = [[]]
        for l in range(len(self.coeffs)):
            if (l == len(self.coeffs)-1):
                print(" Last degree!")
            else:
                eigenfields.append([])
            for m_i in range(len(self.coeffs[l])):
                m = m_i - l
                print(f" l: {l} || m: {m}")
                field = np.zeros((self.y_ind, self.x_ind), dtype=complex)
                for j in range(self.x_ind):
                    for k in range (self.y_ind):
                        x = j
                        y = k
                        theta, phi = self.ct_sp(x,y)
                        field[k,j] = sph_harm(m,l, phi,theta)
                np.nan_to_num(field, copy=False)
                eigenfields[l].append(field)
        self.eigenfields = eigenfields

    def generate_frame(self, radius=1, new_coeffs=False):
        """
        returns a complex np matrix of harmonics at specified
        radius (default = 1) and coefficients (default = self.coeffs)
        """
        localc = self.coeffs
        if new_coeffs != False:
            localc = new_coeffs
        net_field = np.zeros((self.y_ind, self.x_ind), dtype=complex)
        for l in range(len(self.coeffs)):
            for m in range(len(self.coeffs[l])):
                net_field += ((radius ** (-1*l-1))*localc[l][m]*self.eigenfields[l][m])
        return net_field
    
    def show_frame(self, radius=1):
        self.show_field(self.generate_frame(radius))

    def show_movie(self, nb_frames=20):
        self.depth_movie(nb_frames)
        f0, ax = plt.subplots()
        fig = ax.imshow(self.world_img)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        carte = ax.imshow(np.real(self.film_reel[0,:,:]), alpha=0.8, cmap='seismic', vmin=np.min(np.real(self.film_reel)), vmax=np.max(np.real(self.film_reel)), interpolation="none")

        def updateData(frame):
            carte.set_array(np.real(self.film_reel[frame,:,:]))
            ax.set_title(f" Radius = {int(6371*self.radii[frame])} / 6371 km")
            return carte

        anime = animation.FuncAnimation(f0, updateData, blit=False, frames=self.film_reel.shape[0], interval=1000, repeat=True)
        #f0.tight_layout()
        plt.show()
        plt.close()
if __name__ == "__main__":
    FG = main()
