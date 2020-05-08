#!/usr/bin/env python3


import bitarray
import numpy as np
import os
import sys


class IdentityReconstructor():
    def __init__(self, datadir):
        self.datadir = datadir

    def reconstruct(self, uid):
        # Read in data and settings
        binfile = os.path.join(self.datadir, str(uid) + '.bin')
        with open(binfile, 'rb') as f:
            B = f.read()
        namefile = os.path.join(self.datadir, str(uid) + '.name')
        with open(namefile, 'r') as f:
            out = f.read()
        # Check length
        if len(B) == 0:
            print("Empty data!")
            msg = "empty data component"
            ffail = os.path.join(self.datadir, str(uid) + '.fail')
            with open(ffail, 'w') as f:
                f.write(msg)
            return ffail
        if (len(B) - 4) != np.frombuffer(B[:4], dtype='<u4')[0]:
            print("Length of received bytes ({}) != length header ({})".format(
                len(B) - 4, np.frombuffer(B[:4], dtype='<u4')[0]))
            msg = "length mismatch"
            ffail = os.path.join(self.datadir, str(uid) + '.fail')
            with open(ffail, 'w') as f:
                f.write(msg)
            return ffail
        # Write output
        fout = os.path.join(self.datadir, out)
        with open(fout, 'wb') as f:
            # Drop initial uint containing bit length
            f.write(B[4:])
        return fout

    def __repr__(self):
        return "IdentityReconstructor: data={}".format(self.datadir)


def main():
    datadir = sys.argv[1]
    uid = sys.argv[2]
    IR = IdentityReconstructor(datadir)
    fname = IR.reconstruct(uid)
    print("Reconstructed", fname)
    return 0


if __name__ == "__main__":
    main()

