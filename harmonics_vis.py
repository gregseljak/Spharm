#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

################################################
###     ------     USER README    ------     ###
###                                          ###
### The field can be manipulated through the ###
### choice of coefficients in this list      ###
### (Under construction, but that's the idea)###
################################################
coeffs = [[1]]

world_img = mpimg.imread("./worldmap.png")
y_ind, x_ind = world_img.shape[0], world_img.shape[1]
#%%
def sp_ct(theta, phi):
    """if ((theta.any() < 0) or (theta.any() > np.pi)):
        print("coordinates are confusing, please respect mine")
        return 0"""
    x_proj = np.sin(theta)*(np.sin(phi)*x_ind/2)+(x_ind/2)
    #x_proj = np.int(x_proj)
    y_proj = (np.cos(theta)+1)*(y_ind/2)
    #y_proj = np.int(y_proj)
    return (x_proj, y_proj)
#%%
def ct_sp(x,y):
    theta = np.arccos(2*y/y_ind)
    
    if (np.abs((2*x/x_ind-1)/np.sin(theta)) <= 1):
        phi = np.arcsin((2*x/x_ind-1)/np.sin(theta))
    else:
        phi = np.nan
    return(theta, phi)

#contour = sp_ct(np.linspace(0,np.pi,y_ind), np.zeros(y_ind)-np.pi/2)
#contour = sp_ct(np.zeros(50)-np.pi/4, np.linspace(-np.pi,np.pi,50))

#%%

def show_field(field=np.zeros((y_ind,x_ind))):
    fig = plt.imshow(world_img)
    test_line = (np.zeros(10), np.linspace(0,np.pi,10))
    #plt.scatter(sp_ct(theta,phi)[0],sp_ct(theta,phi)[1], s=35, color="red")
    plt.imshow(field, alpha=0.2)
    #plt.scatter(*contour, s=0.2)
    #plt.scatter(*(sp_ct(*ct_sp(50,-200))))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()

#%%
def show_point(pointx, pointy):
    fig = plt.imshow(world_img)
    plt.scatter(pointx, pointy, s=2)
    plt.show()
#%%
def draw_field():
    flat = np.zeros((y_ind,x_ind))
    for j in range(x_ind):
        c = 0
        for k in range (y_ind):
            x = j
            y = k - int(y_ind/2)
            theta, phi = ct_sp(x,y)
            if (phi > -1000):
                flat[k,j]=10    # THIS IS WHERE SPHHARM GOES
    show_field(flat)

# %%
def coeff_to_harm(coeffs=[[0]], radius=1):
    l_max = len(coeffs)

    