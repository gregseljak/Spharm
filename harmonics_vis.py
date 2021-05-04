
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.special import sph_harm

#################################################
###     ------     USER README    ------      ###
###                                           ###
### The field can be manipulated through the  ###
### choice of coefficients in this list       ###
### (Under construction, but that's the idea) ###
#################################################
coeffs = [[0],[200 + 200j,0]]
ext_coeffs =[[0]]

world_img = mpimg.imread("./worldmap.png")
y_ind, x_ind = world_img.shape[0], world_img.shape[1]

def sp_ct(theta, phi):
    x_proj = np.sin(theta)*(np.sin(phi)*x_ind/2)+(x_ind/2)
    y_proj = (np.cos(theta)+1)*(y_ind/2)
    return (x_proj, y_proj)

def ct_sp(x,y):
    theta = np.arccos(2*y/y_ind)    
    if (np.abs((2*x/x_ind-1)/np.sin(theta)) <= 1):
        phi = np.arcsin((2*x/x_ind-1)/np.sin(theta))
    else:
        phi = np.nan
    return(theta, phi)

def show_field(field=np.zeros((y_ind,x_ind))):
    fig = plt.imshow(world_img)
    test_line = (np.zeros(10), np.linspace(0,np.pi,10))
    plt.imshow(field, alpha=0.2)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()

def show_point(pointx, pointy):
    fig = plt.imshow(world_img)
    plt.scatter(pointx, pointy, s=2)
    plt.show()

def coeff_harm(theta, phi, coeffs=[[0]], radius=1):
    value = 0
    l_max = len(coeffs)     # ragged list of lists of coefficients
    for l in range(l_max):
        m_max = len(coeffs[l])
        for m in range(m_max):
            value += radius*(-1*l-1)*coeffs[l][m]*sph_harm(m,l,theta,phi)
    return value

def draw_field():
    flat = np.zeros((y_ind,x_ind))
    for j in range(x_ind):
        c = 0
        for k in range (y_ind):
            x = j
            y = k - int(y_ind/2)
            theta, phi = ct_sp(x,y)
            if (phi > -1000):
                flat[k,j] = coeff_harm(theta, phi, coeffs, 1)
    show_field(flat)
    return flat
#flat = draw_field()

def indiv_harms(coeffs=[[0]], radius=1):
    eigenfields = [[]]
    for l in range(len(coeffs)):
        if (l == len(coeffs)-1):
            print(" Last degree!")
        else:
            eigenfields.append([])
        for m_i in range(len(coeffs[l])):
            m = m_i - l
            print(f" l: {l} || m: {m}")
            field = np.zeros((y_ind,x_ind))
            for j in range(x_ind):
                for k in range (y_ind):
                    x = j
                    y = k - int(y_ind/2)
                    theta, phi = ct_sp(x,y)
                    field[k,j] = radius*(-1*l-1)*coeffs[l][m_i]*sph_harm((m),l,theta,phi)
                    eigenfields[l].append(field)
        
    return eigenfields
indies = indiv_harms([[0],[1,0,2]])