from splipy import *
from splipy.IO import *
import splipy.curve_factory as cf
import splipy.surface_factory as sf
import splipy.volume_factory as vf
import numpy as np

# make the star multipatch test case

s1 = sf.square()
s1[0,1] = [  np.cos(np.pi/3),   np.sin(np.pi/3)];
s1[1,1] = [2*np.cos(np.pi/6), 2*np.sin(np.pi/6)];
star = [s1]
for i in range(5):
    star.append(star[-1].clone().rotate(np.pi/3))
for s in star:
  s.raise_order(1,1)
  s.refine(3,3)
with G2("star.g2") as f:
    f.write(star)


# make the L-shape multipatch test case

s1 = sf.square()
s2 = s1 + [1,0]
s3 = s2 - [0,1]
s3.raise_order(0,1) # !!! Different polynomial order for this patch
lshape = [s1,s2,s3]
for surf in lshape:
  surf.refine(5,5)
with G2("lshape.g2") as f:
    f.write(lshape)


# make the self-loop test

s1 = sf.square()
b1 = BSplineBasis(2)
b2 = BSplineBasis(5)
s2 = Surface(b2,b1, [[1,0], [3,0], [3,3], [0,3], [0,1], [1,1], [2,1], [2,2], [1,2], [1,1]])
with G2("self-loop.g2") as f:
    f.write([s1,s2])


# test non-matching parametrization domains
s1 = sf.square()
s2 = s1 + [1,0]
s2.reparam((3,7), (-1,3))
with G2("diff-param-domain.g2") as f:
    f.write([s1,s2])

