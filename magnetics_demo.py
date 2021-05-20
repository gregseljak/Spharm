import numpy as np
import harmonics_vis as field_generator

# This file demonstrates various methods of the FieldGenerator class.
# Please refer to field_generator.py for more details.

FG = field_generator.FieldGenerator()

# update coefficients:
FG.coeffs = [[3], [0.1 + 0.1j, 0, 0]]
FG.show_movie()

# Show a frame of the field at given radius (radius of Earth = 1)
# (Default: r = 6371 km, FG.coeffs)
FG.show_frame(1)

