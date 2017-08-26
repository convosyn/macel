import numpy as np
p = [x for x in input().split('\n\r')]
print(p)
pmb = p[0]
pab = p[1]
pl = p[2]
fp = np.around(pl * (pmb * (1 - pab) + pab * (1 - pmb)), 6)
print(fp)