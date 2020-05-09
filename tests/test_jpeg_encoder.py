import os, sys
sys.path.append(os.path.abspath('..'))
from encoder import *
from PIL import Image
from pprint import pprint

encoder = JPEGEncoder(xfrm='dct')

# Load TIFF image
img = Image.open("NetaLi_small.tiff")
img = np.array(img)[:,:,0:3]
M, N = img.shape[0:2]

# Display image
# plt.figure(figsize=(10,10))
# plt.imshow(img), plt.xticks([]), plt.yticks([])
# plt.show(block=False)

jpeg = encoder.enctopy(img)
# encoder.enctofile(img, "test.jpeg123")
# print(*jpeg)
jpeg_img = encoder.decode(*jpeg)
# filu = encoder.decfromfile("test.jpeg123")

plt.figure(figsize=(10,10))
plt.imshow(jpeg_img.astype(np.uint8)), plt.xticks([]), plt.yticks([])
plt.show()