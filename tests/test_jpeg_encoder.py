import os, sys
sys.path.append(os.path.abspath('..'))
from encoder import *
from PIL import Image
from pprint import pprint
from Helper_functions import psnr
import csv

with open("file_PSNR_table.csv") as table:
    reader = csv.reader(table)
    file_PSNR = {int(row[0]):float(row[1]) for row in reader}
with open("py_PSNR_table.csv") as table:
    reader = csv.reader(table)
    python_PSNR = {int(row[0]):float(row[1]) for row in reader}

print(file_PSNR)
print(python_PSNR)

def test_image_encoding_and_decoding_from_file_dct():
    quality = np.r_[5:101:5]
    encoder = JPEGEncoder(xfrm='dct')
    check = []
    # Load TIFF image
    img = Image.open("NetaLi_small.tiff")
    img = np.array(img)[:, :, 0:3]
    for q in quality:
        q = int(q)
        encoder.enctofile(img, "test.jpeg123", quality=q)
        jpeg_img = encoder.decfromfile("test.jpeg123")
        _psnr = psnr(img, jpeg_img)
        check.append(np.isclose(_psnr, file_PSNR[q], atol=1e-3))

    return all(check)

def test_image_encoding_and_decoding_from_python_dct():
    quality = np.r_[5:101:5]
    encoder = JPEGEncoder(xfrm='dct')
    check = []
    # Load TIFF image
    img = Image.open("NetaLi_small.tiff")
    img = np.array(img)[:, :, 0:3]
    for q in quality:
        q = int(q)
        jpeg = encoder.enctopy(img, quality=q)
        jpeg_img = encoder.decode(*jpeg)
        _psnr = psnr(img, jpeg_img)
        check.append(np.isclose(_psnr, python_PSNR[q], atol=1e-3))

    return all(check)



# if __name__ == "__main__":
#     with open("file_PSNR_table.csv", 'w') as file:
#         for quality in np.r_[5:101:5]:
#             quality = int(quality)
#             _psnr = test_image_encoding_and_decoding_from_file_dct(quality=quality)
#             file.write(f"{quality}, {_psnr}\n")
#     with open("py_PSNR_table.csv", 'w') as file:
#         for quality in np.r_[5:101:5]:
#             quality = int(quality)
#             _psnr = test_image_encoding_and_decoding_from_python_dct(quality=quality)
#             file.write(f"{quality}, {_psnr}\n")