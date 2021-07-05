
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
### e.x. C(l=3, m=-1) = coeffs[3][2]          ###
###                                           ###
########################## Greg Seljak, 2021  ###
#################################################
coeffs = [[0], [0, 1, 1], [0,0,0,0,2],[0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]]


##################################################
# You don't need to rerun the program each time! #
# Run this file is in interactive console:       #
#   $python3 -i field_generator.py               #
#                                                #
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
    try:
        FG.show_movie()
    except IndexError:
        return FG
    return FG


class FieldGenerator(object):

    def __init__(self, imagepath="./worldmap.png"):
        self.insufficient_fields = False
        self.world_img = mpimg.imread(imagepath)
        self.y_ind, self.x_ind = self.world_img.shape[0], self.world_img.shape[1]
        self._coeffs = coeffs
        self.globalmax = 0
        self.globalmin = 0
        try:
            self.load_eigenfields()
        except FileNotFoundError:
            print(" ./fieldvalues.npz not found (normal for first-time setup)")
            print(" Generating field values. This can take a while:")
            self.generate_eigenfields()
            self.save_basis()
        self._check_formatting()


    @property
    def coeffs(self):
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value):
        self._coeffs = value
        self._check_formatting()

    def _check_formatting(self):
        if (len(self.eigenfields) >= len(self._coeffs)):
            for l in range(len(self._coeffs)):
                if len(self.eigenfields[l]) == len(self._coeffs[l]) and len(self._coeffs[l]) == 2*l+1:
                    for m in range(len(self._coeffs[l])):
                        if type(self._coeffs[l][m]) in (int, float, complex):
                            continue
                        else:
                            print(
                                f" coeffs list formatting error; uneven list heirarchy at l={l}, m={m}")
                            print(type(self._coeffs[l][m]))
                else:
                    print(
                        f" coeffs list formatting error; bad list length at l={l}")
        else:
            print(
                f" coeffs list formatting error; more coefficients than generated fields.")
            print(f" Delete the ./fieldvalues.npz file and then run FG.save_basis() to extend the generated list!")
            print(" (>>> FG.save_basis() in main() or in console)")
            self.insufficient_fields = True

    def save_basis(self):
        if self.insufficient_fields == True:
            self.generate_eigenfields()
        field_archive = []
        for degree in self.eigenfields:
            for field in degree:
                field_archive.append(field)
        np.savez("fieldvalues", field_archive)

    def load_eigenfields(self):
        """ pull the fields from the fieldvalues.npz file"""
        field_archive = np.load("./fieldvalues.npz")['arr_0.npy']
        self.eigenfields = []
        deg = np.sqrt(field_archive.shape[0])
        if deg != int(deg):
            print(
                f" Discrepancy; loaded file has nonsquare # of fields ({field_archive.shape[0]})")
            return " error; corrupted ./fieldvalues.npz; Delete it and restart"
        idx = 0
        for l in range(int(deg)):
            self.eigenfields.append([])
            for m in range(2*l+1):
                self.eigenfields[l].append(field_archive[idx])
                idx += 1

    def _depth_movie(self, nb_frames):
        """ create the frames for the movie and store them in self.film_reel"""
        self.radii = np.linspace(0.5, 1.5, nb_frames)
        film_reel = np.zeros(
            (nb_frames, self.y_ind, self.x_ind), dtype=complex)
        for i in range(nb_frames):
            film_reel[i, :, :] = self.generate_frame(self.radii[i])
        self.film_reel = film_reel

    def _ct_sp(self, x, y):
        # Certainly the most difficult part of the project!
        # cartesian (pixel) to spherical (lat, lon) coordinate
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
        """ show the field argument over the bacckground image """
        realfield = np.real(field)
        fig = plt.imshow(self.world_img)
        plt.imshow(realfield, alpha=0.8, cmap='seismic', vmin=np.min(
            realfield), vmax=np.max(realfield), interpolation="none")
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.show()

    def _show_point(self, pointx, pointy):
        """ debug function """
        fig = plt.imshow(self.world_img)
        plt.scatter(pointx, pointy, s=5)
        plt.show()

    def generate_eigenfields(self):
        """
        Generate a ragged list of complex-valued fields over the image.
        Fields are generated up to degree L = len(FG.coeffs) = len(coeffs)
        Computationally intensive, but it only has to be done once!
        """
        eigenfields = [[]]
        for l in range(len(self.coeffs)):
            if (l == len(self.coeffs)-1):
                print(" Last degree! almost done")
            else:
                eigenfields.append([])
            for m_i in range(len(self.coeffs[l])):
                m = m_i - l
                print(f" l: {l} || m: {m}   ")
                field = np.zeros((self.y_ind, self.x_ind),
                                 dtype=complex) + 10*l + m_i
                field = np.zeros((self.y_ind, self.x_ind), dtype=complex)
                for j in range(self.x_ind):
                    for k in range(self.y_ind):
                        x = j
                        y = k
                        theta, phi = self._ct_sp(x, y)
                        field[k, j] = sph_harm(m, l, phi, theta)
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
                net_field += ((radius ** (-1*l-1)) *
                              localc[l][m]*self.eigenfields[l][m])
        return net_field

    def show_frame(self, radius=1):
        self.show_field(self.generate_frame(radius))

    def show_movie(self, nb_frames=20):
        self._depth_movie(nb_frames)
        f0, ax = plt.subplots()
        fig = ax.imshow(self.world_img)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        carte = ax.imshow(np.real(self.film_reel[0, :, :]), alpha=0.8, cmap='seismic', vmin=np.min(
            np.real(self.film_reel)), vmax=np.max(np.real(self.film_reel)), interpolation="none")

        def updateData(frame):
            carte.set_array(np.real(self.film_reel[frame, :, :]))
            ax.set_title(f" Radius = {int(6371*self.radii[frame])} / 6371 km")
            return carte

        anime = animation.FuncAnimation(
            f0, updateData, blit=False, frames=self.film_reel.shape[0], interval=1000, repeat=True)
        # f0.tight_layout()
        plt.show()
        plt.close()


if __name__ == "__main__":
    FG = main()
