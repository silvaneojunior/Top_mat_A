from sympy import beta
import tensorflow as tf
import tensorflow.keras as k
import numpy as np
from numpy import pi

import API_Numpy
import API_TensorFlow

float_pres='float64'

def const(x, API):
    return API.constant(x, dtype = float_pres)

ɛ = np.asarray(10, dtype=float_pres)**np.asarray(-40, dtype=float_pres)

B = np.asarray([[1,0,0],[0,6,0],[0,0,3]], dtype=float_pres)/10                # Matriz B
C = np.asarray([[2,-7,11,0,0],[0,-1,5,2,0],[0,0,2,5,-1]], dtype=float_pres)/6 # Matriz C
C = np.transpose(C)

γ = np.asarray(14, dtype=float_pres)/10