#!/usr/bin/env python3


import bitarray
import numpy as np
import os
import sys


class IdentityReconstructor():
    def __init__(self, datadir):
        self.datadir = datadir

    def reconstruct(self, binfile, outfile):
        # Read in data and settings
        with open(binfile, 'rb') as f:
            B = f.read()
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
        with open(outfile, 'wb') as f:
            # Drop initial uint containing bit length
            f.write(B[4:])
        return outfile

    def __repr__(self):
        return "IdentityReconstructor: data={}".format(self.datadir)


def main():
    datadir = sys.argv[1]
    binfile = sys.argv[2]
    outfile = sys.argv[3]
    IR = IdentityReconstructor(datadir)
    fname = IR.reconstruct(binfile, outfile)
    print("Reconstructed", fname)
    return 0


if __name__ == "__main__":
    main()

