import os, sys, shutil
sys.path.append(os.path.abspath(".."))
from encoder import *
from Helper_functions import *
import pathlib
import matplotlib.pyplot as plt
import glob

test_vid = pathlib.Path(os.path.abspath(".")).parent / "example_videos" / "simple_shape.png"

img_stack = imageStack_load(str(test_vid))

encoder = JPEGEncoder(quality=25, xfrm='dct')
drop = test_vid.parent / "test_frames"
shutil.rmtree(drop)
os.mkdir(drop)


i = 1
for img in img_stack:
    # plt.figure(figsize=(15, 15))
    plt.imshow(img), plt.xticks([]), plt.yticks([])
    plt.savefig(drop / f"frame{i}.tiff")
    fpath = drop / f"frame{i}.jpeg123"
    encoder.enctofile(img, fpath)
    i += 1

os.chdir(drop)
frames = glob.glob("*.jpeg123")

i = 1
print(frames)
for frame in frames:
    name = frame.split(".")[0]
    fpath = drop / f"{name}.png"
    img = encoder.decfromfile(frame)
    # plt.figure(figsize=(15, 15))
    plt.imshow(img), plt.xticks([]), plt.yticks([])
    plt.savefig(fpath)
    i += 1



