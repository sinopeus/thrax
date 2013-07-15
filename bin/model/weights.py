from math import pow, sqrt
import numpy.random

def random_weights(nin, nout, scale_by=1./sqrt3, power=0.5):
    sqrt3 = sqrt(3.0)
    return (numpy.random.rand(nin, nout) * 2.0 - 1) * scale_by * sqrt3 / pow(nin,power) 
