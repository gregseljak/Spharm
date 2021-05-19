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
#################################################
#################################################

# N.B. To comply with scipy's convention, I follow this convention:
# (theta, phi)                     # (0,0)
#  N,S,E,W :     (pi/2, -pi) #         # (pi/2, pi)
#                         (pi, 0) #
################################################

class FieldGenerator(object):

    def __init__(self, imagepath="./worldmap.png"):
        self.world_img = mpimg.imread(imagepath)
        self.y_ind, self.x_ind = self.world_img.shape[0], self.world_img.shape[1]
        self.coeffs = coeffs

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

    def show_field(self, field=np.zeros((self.y_ind,self.x_ind))):
        fig = plt.imshow(self.world_img)
        plt.imshow(field, alpha=0.8, cmap='gray', vmin=np.min(field), vmax=np.max(field), interpolation="none")
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.show()

    def show_point(self, pointx, pointy):
        fig = plt.imshow(self.world_img)
        plt.scatter(pointx, pointy, s=5)
        plt.show()

    def coeff_harm(self, theta, phi, coeffs=[[0]], radius=1):
        value = 0
        l_max = len(coeffs)     # ragged list of lists of coefficients
        for l in range(l_max):
            m_max = len(coeffs[l])
            for m in range(m_max):
                value += radius*(-1*l-1)*coeffs[l][m]*sph_harm(m,l,theta,phi)
        return value

    def draw_field(self):
        flat = np.zeros((self.y_ind, self.x_ind))
        for j in range(self.x_ind):
            c = 0
            for k in range (self.y_ind):
                x = j
                y = k - int(self.y_ind/2)
                theta, phi = self.ct_sp(x,y)
                if (phi > -1000):
                    flat[k,j] = self.coeff_harm(theta, phi, coeffs, 1)
        #self.show_field(flat)
        return flat
#flat = draw_field()

    def indiv_harms(self, radius=1):
        eigenfields = [[]]
        for l in range(len(self.coeffs)):
            if (l == len(self.coeffs)-1):
                print(" Last degree!")
            else:
                eigenfields.append([])
            for m_i in range(len(self.coeffs[l])):
                m = m_i - l
                print(f" l: {l} || m: {m}")
                field = np.zeros((self.y_ind, self.x_ind))
                for j in range(self.x_ind):
                    if self.coeffs[l][m_i] == 0:
                        continue
                    for k in range (self.y_ind):
                        x = j
                        y = k
                        theta, phi = self.ct_sp(x,y)
                        field[k,j] = np.real(radius*(-1*l-1)*coeffs[l][m_i]*sph_harm(m,l, phi,theta))
                np.nan_to_num(field, copy=False)
                eigenfields[l].append(field)
            
        return eigenfields

    def generate_frame(self, eigenfields, radius=1):
        net_field = np.zeros((self.y_ind, self.x_ind))
        for l in range(len(eigenfields)):
            for m in range(len(eigenfields[l])):
                net_field += radius ** (-1*l-1)
        return net_field

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
