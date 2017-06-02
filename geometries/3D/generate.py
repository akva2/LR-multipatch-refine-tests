from splipy import *
from splipy.IO import *
import splipy.curve_factory as cf
import splipy.surface_factory as sf
import splipy.volume_factory as vf
import numpy as np

# 3-patch Lshape example (xz-plane) extruded in y-direction
b1 = Volume()
b2 = Volume() + [1,0,0]
b3 = Volume() + [0,0,1]
vols = [b1,b2,b3]
for b in vols:
  b.raise_order(1,1,1)
  b.refine(3,3,3)
with G2("boxes-3.g2") as f:
  f.write(vols)

# 3-patch circular staircase
c1 = cf.circle_segment(theta=np.pi/4, r=1)
c2 = c1*2
surf = sf.edge_curves(c1,c2)
surf.raise_order(0,1)
surf.refine(3,3)
surf.swap()
v1 = vf.extrude(surf, [0,0,.5])
v1 = v1.rebuild(p=(3,3,3), n=(6,6,6)) # rebuild so it is not rational
v2 = v1.clone().rotate(theta=np.pi/4)
v3 = v2 + [0,0,.5]
with G2("disc-stair-3.g2") as f:
  f.write([v1,v2,v3])

# 10-patch version of the above example
v4  = v2.clone().rotate(theta=np.pi/4)
v5  = v4 + [0,0,.5]
v6  = v5 + [0,0,.5]
v7  = v4.clone().rotate(theta=np.pi/4)
v8  = v7 + [0,0,.5]
v9  = v8 + [0,0,.5]
v10 = v9 + [0,0,.5]
with G2("disc-stair-10.g2") as f:
  f.write([v1,v2,v3,v4,v5,v6,v7,v8,v9,v10])

# 7-unit cubes in all 8 quadrants around the origin, save one where x>0,y>0,z>0
v1 = Volume()
vol = [v1]
for i in range(3):
  vol.append(vol[-1].clone().rotate(np.pi/2))
for i in range(4):
  vol.append(vol[i].clone().mirror([0,0,1]))
  vol[-1].swap(0,1)
for v in vol:
  v.raise_order(2,2,2)
  v.refine(1,1,1)

with G2("boxes-7.g2") as f:
  f.write(vol[1:])


