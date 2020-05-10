import os, sys

sys.path.append(os.path.abspath(".."))
from wavelets import *


def test_haar_wavelet_generation():
    # Test Haar-4
    golden_haar = {'father': np.ones(8) * (1 / np.sqrt(8)),
                   (2, 0): np.asarray([-1, -1, -1, -1, 1, 1, 1, 1]) * (1 / np.sqrt(8)),
                   (1, 0): np.asarray([-1, -1, 1, 1, 0, 0, 0, 0]) * 0.5,
                   (1, 1): np.asarray([0, 0, 0, 0, -1, -1, 1, 1]) * 1 / np.sqrt(2),
                   (0, 0): np.asarray([-1, 1, 0, 0, 0, 0, 0, 0]) * 1 / np.sqrt(2),
                   (0, 1): np.asarray([0, 0, -1, 1, 0, 0, 0, 0]) * 1 / np.sqrt(2),
                   (0, 2): np.asarray([0, 0, 0, 0, -1, 1, 0, 0]) * 1 / np.sqrt(2),
                   (0, 3): np.asarray([0, 0, 0, 0, 0, 0, -1, 1]) * 1 / np.sqrt(2),
                   }

    haar = Haar(2, 8)
    family = haar.family
    equal = []
    for key, value in golden_haar.items():
        equal.append((key,all(np.isclose(value , family[key]))))
    print(equal)
    assert(all(equal))

haar = Haar(2, 8)
# test_haar_wavelet_generation()
print(haar(0,2))
print(haar.get_xfrm_mat())