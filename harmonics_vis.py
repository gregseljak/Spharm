#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.special import sph_harm

#################################################
###     ------     USER README    ------      ###
###                                           ###
### The field can be manipulated through the  ###
### choice of coefficients in this list       ###
### e.g. C(l=2, m=-1) = coeffs[2][2]          ###
#################################################

coeffs = [[0],[1,1,1],[0,0,0,0,0]]
ext_coeffs =[[0]]
#%%
##################################################
# You don't need to rerun the program each time! #
# Run this is in interactive console:            #
#   $python3 -i harmonics_vis.py                 #
#                                                #
# Now you can manipulate and re-run the visuals  #
# without having to recompute the fields:        #
#                                                #
# >>> FG.coeffs = [[0.4], [0+2j,0.1,1]]          #
# >>> FG.see_frame()                             #
# >>> FG.depth_movie()                           #
##################################################
#
# N.B. To comply with scipy's convention, I follow this convention:
# (theta, phi)                     # (0,0)
#  N,S,E,W :     (pi/2, -pi) #         # (pi/2, pi)
#                         (pi, 0) #
#
################################################

def main():
    FG = FieldGenerator()
    FG.show_field(FG.generate_frame())
    return FG

class FieldGenerator(object):

    def __init__(self, imagepath="./worldmap.png"):
        self.world_img = mpimg.imread(imagepath)
        self.y_ind, self.x_ind = self.world_img.shape[0], self.world_img.shape[1]
        self.coeffs = coeffs
        self.generate_eigenfields()

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
        fig = plt.imshow(self.world_img)
        plt.imshow(field, alpha=0.8, cmap='gray', vmin=np.min(field), vmax=np.max(field), interpolation="none")
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
                    if self.coeffs[l][m_i] == 0:
                        continue
                    for k in range (self.y_ind):
                        x = j
                        y = k
                        theta, phi = self.ct_sp(x,y)
                        field[k,j] = sph_harm(m,l, phi,theta)
                np.nan_to_num(field, copy=False)
                eigenfields[l].append(field)
        self.eigenfields = eigenfields

    def generate_frame(self, radius=1):
        net_field = np.zeros((self.y_ind, self.x_ind))
        for l in range(len(self.coeffs)):
            for m in range(len(self.coeffs[l])):
                net_field += np.real((radius ** (-1*l-1))*coeffs[l][m]*self.eigenfields[l][m])
        return net_field
    
    def see_frame(self):
        self.show_field(self.generate_frame())
FG = main()

"""
indies1 = indiv_harms(coeffs)
#%%
indies2 = indiv_harms([[0], [1j,1j,1]])
#%%
show_field(indies1[1][0])
show_field(indies2[1][2])

# %%
def tester(x,y):
    #show_point(x,y)
    return(ct_sp(x,y))
"""
