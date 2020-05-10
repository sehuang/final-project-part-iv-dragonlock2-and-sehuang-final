import numpy as np
import scipy.fftpack
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from bitarray import bitarray
from huffman import *
from os import stat
from PIL import Image
from enum import Enum
from wavelets import *


class ValidTransforms(Enum):
    dct = 2
    DCT = 2
    dct2 = 2
    DCT2 = 2
    wavelet = 1


class JPEGEncoder:
    def __init__(self, quality=75, xfrm='wavelet'):
        self.quality = quality
        try:
            self.xfrm = ValidTransforms[xfrm]
        except KeyError:
            print(f"{xfrm} is not a valid transform type.")

        # Constants
        # =========
        # RGB to YCbCr Transformation matrix
        self.scale_mat = np.asarray([[0.299, 0.587, 0.114],
                                     [-0.1687360, -0.331264, 0.5],
                                     [0.5, -0.418688, -0.081312]])

        # YCbCr to RGB Transformation matrix
        self.unscale_mat = np.linalg.inv(self.scale_mat)

        # Block length
        self.length = 64  # NOTE: This must be a perfect square!

        # Luma quantization array
        self.Qy_arr = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 36, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])

        # Chroma quantization array
        self.Qc_arr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                [18, 21, 26, 66, 99, 99, 99, 99],
                                [24, 26, 56, 99, 99, 99, 99, 99],
                                [47, 66, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99]])

        # Zig-zag encoding indices
        self.zz_idx = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4,
                       5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7,
                       14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22,
                       15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39,
                       46, 53, 60, 61, 54, 47, 55, 62, 63]

        # Zig-zag decoding indices
        self.uzz_idx = [0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26,
                        29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11, 18,
                        24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54,
                        20, 22, 33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50,
                        56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63]

    def RGB2YCbCr(self, im_rgb):
        # Input:  a 3D float array, im_rgb, representing an RGB image in range [0.0,255.0]
        # Output: a 3D float array, im_ycbcr, representing a YCbCr image in range [-128.0,127.0]

        # -128 for all Y
        const = np.asarray(
            [[np.ones(im_rgb.shape[0]) * -128],
             [np.zeros(im_rgb.shape[0])],
             [np.zeros(im_rgb.shape[0])]]).T

        # Matmul and add constant
        im_ycbcr = np.matmul(self.scale_mat, im_rgb.transpose(1, 2, 0)).transpose(2, 0, 1) + const
        return im_ycbcr

    def YCbCr2RGB(self, im_ycbcr):
        # Input:  a 3D float array, im_ycbcr, representing a YCbCr image in range [-128.0,127.0]
        # Output: a 3D float array, im_rgb, representing an RGB image in range [0.0,255.0]

        # Your code here
        const = np.asarray(
            [[np.ones(im_ycbcr.shape[0]) * 128],
             [np.zeros(im_ycbcr.shape[0])],
             [np.zeros(im_ycbcr.shape[0])]]).T

        im_ycbcr = im_ycbcr + const
        im_rgb = np.matmul(self.unscale_mat, im_ycbcr.transpose(1, 2, 0)).transpose(2, 0, 1)
        # Clip RGB output
        im_rgb = np.where(im_rgb > 255.0, 255.0, im_rgb)
        im_rgb = np.where(im_rgb < 0.0, 0.0, im_rgb)

        return im_rgb

    def chroma_downsample(self, C):
        # Input:  an MxN array, C, of chroma values
        # Output: an (M/2)x(N/2) array, C2, of downsampled chroma values

        # Your code here
        Image_C = Image.fromarray(C)
        size = Image_C.size
        C2 = np.array(Image_C.resize((size[0] // 2, size[1] // 2), resample=Image.BILINEAR))

        return C2

    def chroma_upsample(self, C2):
        # Input:  an (M/2)x(N/2) array, C2, of downsampled chroma values
        # Output: an MxN array, C, of chroma values

        # Your code here
        Image_C = Image.fromarray(C2)
        size = Image_C.size
        C = np.array(Image_C.resize((size[0] * 2, size[1] * 2), resample=Image.BILINEAR))

        return C

    def _get_xfrm(self):
        if self.xfrm.value == 1:
            return self.wvlt
        elif self.xfrm.value == 2:
            return self.dct2

    def _get_ixfrm(self):
        if self.xfrm.value == 1:
            return self.iwvlt
        elif self.xfrm.value == 2:
            return self.idct2

    def dct2(self, block):
        # Input:  a 2D array, block, representing an image block
        # Output: a 2D array, block_c, of DCT coefficients

        # Your code here
        block_c = scipy.fftpack.dct(scipy.fftpack.dct(block, norm="ortho").T, norm="ortho").T

        return block_c

    def idct2(self, block_c):
        # Input:  a 2D array, block_c, of DCT coefficients
        # Output: a 2D array, block, representing an image block

        # Your code here
        block = scipy.fftpack.idct(scipy.fftpack.idct(block_c.T, norm="ortho").T, norm="ortho")

        return block

    def wvlt(self, block):
        pass

    def iwvlt(self, block):
        pass

    def quantize(self, block_c, mode="y", quality=75):
        # Input:  a 2D float array, block_c, of DCT coefficients
        #         a string, mode, ("y" for luma quantization, "c" for chroma quantization)
        #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
        # Output: a 2D int array, block_cq, of quantized DCT coefficients

        if mode is "y":
            Q = self.Qy_arr
        elif mode is "c":
            Q = self.Qc_arr
        else:
            raise Exception("String argument must be 'y' or 'c'.")

        if quality < 1 or quality > 100:
            raise Exception("Quality factor must be in range [1,100].")

        scalar = 5000 / quality if quality < 50 else 200 - 2 * quality  # formula for scaling by quality factor
        Q = Q * scalar / 100.  # scale the quantization matrix
        Q[Q < 1.] = 1.  # do not divide by numbers less than 1

        # Your code here
        block_cq = np.round(np.divide(block_c, Q))

        return block_cq

    def unquantize(self, block_cq, mode="y", quality=75):
        # Input:  a 2D int array, block_cq, of quantized DCT coefficients
        #         a string, mode, ("y" for luma quantization, "c" for chroma quantization)
        #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
        # Output: a 2D float array, block_c, of "unquantized" DCT coefficients (they will still be quantized)

        if mode is "y":
            Q = self.Qy_arr
        elif mode is "c":
            Q = self.Qc_arr
        else:
            raise Exception("String argument must be 'y' or 'c'.")

        if quality < 1 or quality > 100:
            raise Exception("Quality factor must be in range [1,100].")

        scalar = 5000 / quality if quality < 50 else 200 - 2 * quality  # formula for scaling by quality factor
        Q = Q * scalar / 100.  # scale the quantization matrix
        Q[Q < 1.] = 1.  # do not divide by numbers less than 1

        # Your code here
        block_c = np.multiply(block_cq, Q)

        return block_c

    def zrle(self, block):
        block_1d = np.ravel(block).tolist()
        # Reindex
        block_cqz = [block_1d[i] for i in self.zz_idx]
        block_cqzr = [block_cqz[0]]
        count = 0
        idx = 0
        for el in block_cqz[1:]:
            if el == 0:
                count += 1
                # If encountered run of 16 0s before next nonzero
                if count == 16:
                    if any(block_cqz[idx:]):
                        count = 0
                        block_cqzr.append((15, 0))
            else:
                block_cqzr.append((count, el))
                count = 0
            idx += 1
        # Make sure there are the right amount of 0 at the end
        block_cqzr.append((0, 0))

        return block_cqzr

    def unzrle(self, block):

        block_cqz = []
        for el in block:
            if isinstance(el, tuple):
                sublist = [0] * el[0] + [el[1]]
                block_cqz += sublist
            else:
                block_cqz.append(el)
        if (self.length - len(block_cqz)) > 0:
            block_cqz += [0] * (self.length - len(block_cqz))

        block_1d = np.asarray([block_cqz[i] for i in self.uzz_idx])
        # Reshape
        r = int(np.sqrt(self.length))
        block_cq = np.reshape(block_1d, (r, r))

        return block_cq

    def encode_block(self, block, mode="y", quality=75):
        # Input:  a 2D array, block, representing an image component block
        #         a string, mode, ("y" for luma, "c" for chroma)
        #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
        # Output: a bitarray, dc_bits, of Huffman encoded DC coefficients
        #         a bitarray, ac_bits, of Huffman encoded AC coefficients

        xfrm = self._get_xfrm()
        block_c = xfrm(block)
        block_cq = self.quantize(block_c, mode, quality)
        block_cqzr = self.zrle(block_cq)
        dc_bits = encode_huffman(block_cqzr[0], mode)  # DC
        ac_bits = ''.join(encode_huffman(v, mode) for v in block_cqzr[1:])  # AC

        return bitarray(dc_bits), bitarray(ac_bits)

    def decode_block(self, dc_gen, ac_gen, mode="y", quality=75):
        # Inputs: a generator, dc_gen, that yields decoded Huffman DC coefficients
        #         a generator, ac_gen, that yields decoded Huffman AC coefficients
        #         a string, mode, ("y" for luma, "c" for chroma)
        #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
        # Output: a 2D array, block, decoded by and yielded from the two generators

        ixfrm = self._get_ixfrm()
        block_cqzr = [next(dc_gen)]  # initialize list by yielding from DC generator
        while block_cqzr[-1] != (0, 0):
            block_cqzr.append(next(ac_gen))  # append to list by yielding from AC generator until (0,0) is encountered
        block_cqz = self.unzrle(block_cqzr)
        block_c = self.unquantize(block_cqz, mode, quality)
        block = ixfrm(block_c)

        return block

    def mirror_pad(self, img):
        # Input:  a 3D float array, img, representing an RGB image in range [0.0,255.0]
        # Output: a 3D float array, img_pad, mirror padded so the number of rows and columns are multiples of 16

        M, N = img.shape[0:2]
        pad_r = ((16 - (M % 16)) % 16)  # number of rows to pad
        pad_c = ((16 - (N % 16)) % 16)  # number of columns to pad
        img_pad = np.pad(img, ((0, pad_r), (0, pad_c), (0, 0)), "symmetric")  # symmetric padding

        return img_pad

    def encode(self, img, quality=None):
        if quality:
            self.quality = quality

        quality = self.quality
        img = self.mirror_pad(img[:, :, 0:3])
        M, N = img.shape[0:2]

        im_ycbcr = self.RGB2YCbCr(img)
        Y = im_ycbcr[:, :, 0]
        Cb = self.chroma_downsample(im_ycbcr[:, :, 1])
        Cr = self.chroma_downsample(im_ycbcr[:, :, 2])

        # Y component
        Y_dc_bits = bitarray()
        Y_ac_bits = bitarray()
        for i in np.r_[0:M:8]:
            for j in np.r_[0:N:8]:
                block = Y[i:i + 8, j:j + 8]
                dc_bits, ac_bits = self.encode_block(block, "y", quality)
                Y_dc_bits.extend(dc_bits)
                Y_ac_bits.extend(ac_bits)

        # Cb component
        Cb_dc_bits = bitarray()
        Cb_ac_bits = bitarray()
        for i in np.r_[0:M // 2:8]:
            for j in np.r_[0:N // 2:8]:
                block = Cb[i:i + 8, j:j + 8]
                dc_bits, ac_bits = self.encode_block(block, "c", quality)
                Cb_dc_bits.extend(dc_bits)
                Cb_ac_bits.extend(ac_bits)

        # Cr component
        Cr_dc_bits = bitarray()
        Cr_ac_bits = bitarray()
        for i in np.r_[0:M // 2:8]:
            for j in np.r_[0:N // 2:8]:
                block = Cr[i:i + 8, j:j + 8]
                dc_bits, ac_bits = self.encode_block(block, "c", quality)
                Cr_dc_bits.extend(dc_bits)
                Cr_ac_bits.extend(ac_bits)

        bits = (Y_dc_bits, Y_ac_bits, Cb_dc_bits, Cb_ac_bits, Cr_dc_bits, Cr_ac_bits)

        return bits

    def decode(self, bits, M, N, quality=75):
        # Inputs: a tuple, bits, containing the following:
        #              a bitarray, Y_dc_bits, the Y component DC bitstream
        #              a bitarray, Y_ac_bits, the Y component AC bitstream
        #              a bitarray, Cb_dc_bits, the Cb component DC bitstream
        #              a bitarray, Cb_ac_bits, the Cb component AC bitstream
        #              a bitarray, Cr_dc_bits, the Cr component DC bitstream
        #              a bitarray, Cr_ac_bits, the Cr component AC bitstream
        #         ints, M and N, the number of rows and columns in the image
        #         an int, quality, the JPEG quality factor in range [1,100] (defaults to 75)
        # Output: a 3D float array, img, representing an RGB image in range [0.0,255.0]

        Y_dc_bits, Y_ac_bits, Cb_dc_bits, Cb_ac_bits, Cr_dc_bits, Cr_ac_bits = bits  # unpack bits tuple

        M_orig = M  # save original image dimensions
        N_orig = N
        M = M_orig + ((16 - (M_orig % 16)) % 16)  # dimensions of padded image
        N = N_orig + ((16 - (N_orig % 16)) % 16)
        num_blocks = M * N // 64  # number of blocks

        # Y component
        Y_dc_gen = decode_huffman(Y_dc_bits.to01(), "dc", "y")
        Y_ac_gen = decode_huffman(Y_ac_bits.to01(), "ac", "y")
        Y = np.empty((M, N))
        for b in range(num_blocks):
            block = self.decode_block(Y_dc_gen, Y_ac_gen, "y", quality)
            r = (b * 8 // N) * 8  # row index (top left corner)
            c = b * 8 % N  # column index (top left corner)
            Y[r:r + 8, c:c + 8] = block

        # Cb component
        Cb_dc_gen = decode_huffman(Cb_dc_bits.to01(), "dc", "c")
        Cb_ac_gen = decode_huffman(Cb_ac_bits.to01(), "ac", "c")
        Cb2 = np.empty((M // 2, N // 2))
        for b in range(num_blocks // 4):
            block = self.decode_block(Cb_dc_gen, Cb_ac_gen, "c", quality)
            r = (b * 8 // (N // 2)) * 8  # row index (top left corner)
            c = b * 8 % (N // 2)  # column index (top left corner)
            Cb2[r:r + 8, c:c + 8] = block

        # Cr component
        Cr_dc_gen = decode_huffman(Cr_dc_bits.to01(), "dc", "c")
        Cr_ac_gen = decode_huffman(Cr_ac_bits.to01(), "ac", "c")
        Cr2 = np.empty((M // 2, N // 2))
        for b in range(num_blocks // 4):
            block = self.decode_block(Cr_dc_gen, Cr_ac_gen, "c", quality)
            r = (b * 8 // (N // 2)) * 8  # row index (top left corner)
            c = b * 8 % (N // 2)  # column index (top left corner)
            Cr2[r:r + 8, c:c + 8] = block

        Cb = self.chroma_upsample(Cb2)
        Cr = self.chroma_upsample(Cr2)

        img = self.YCbCr2RGB(np.stack((Y, Cb, Cr), axis=-1))

        img = img[0:M_orig, 0:N_orig, :]  # crop out padded parts

        return img

    def enctofile(self, source, outfile, quality=None):
        if quality:
            self.quality = quality

        img = source.astype(np.float64)
        M, N = img.shape[0:2]

        bits = self.encode(img, quality=self.quality)
        for barray in bits:
            offset = (16 - (len(barray) % 16)) % 16
            if offset:
                barray.extend([True] * offset)

        soi = bytes.fromhex("ff d8")
        rows = (M).to_bytes(2, "big")
        cols = (N).to_bytes(2, "big")
        q_factor = (self.quality).to_bytes(2, "big")
        sos = bytes.fromhex("ff da")
        (Y_dc, Y_ac, Cb_dc, Cb_ac, Cr_dc, Cr_ac) = bits
        eoi = bytes.fromhex("ff d9")

        with open(outfile, "wb") as fh:
            # Your code here
            fh.write(soi)
            fh.write(rows)
            fh.write(cols)
            fh.write(q_factor)
            fh.write(sos)
            Y_dc.tofile(fh)
            fh.write(sos)
            Y_ac.tofile(fh)
            fh.write(sos)
            Cb_dc.tofile(fh)
            fh.write(sos)
            Cb_ac.tofile(fh)
            fh.write(sos)
            Cr_dc.tofile(fh)
            fh.write(sos)
            Cr_ac.tofile(fh)
            fh.write(eoi)

    def enctopy(self, source, quality=None):
        if quality:
            self.quality = quality

        img = source.astype(np.float64)
        M, N = img.shape[0:2]

        bits = self.encode(img, quality=self.quality)
        for barray in bits:
            offset = (16 - (len(barray) % 16)) % 16
            if offset:
                barray.extend([True] * offset)

        return (bits, M, N, self.quality)

    def decfromfile(self, infile):
        # Inputs:  a string, infile, of the input binary filename
        # Outputs: a 3D uint8 array, img_dec, representing a decoded JPEG123 color image

        with open(infile, "rb") as fh:
            SOI = fh.read(2)
            if SOI != bytes.fromhex("FFD8"):
                raise Exception("Start of Image marker not found!")
            M = int.from_bytes(fh.read(2), "big")
            N = int.from_bytes(fh.read(2), "big")
            quality = int.from_bytes(fh.read(2), "big")
            SOS = fh.read(2)
            if SOS != bytes.fromhex("FFDA"):
                raise Exception("Start of Scan marker not found!")
            bits = ()
            for _ in range(5):
                ba = bitarray()
                for b in iter(lambda: fh.read(2), bytes.fromhex("FFDA")):  # iterate until next SOS marker
                    ba.frombytes(b)
                bits = (*bits, ba)
            ba = bitarray()
            for b in iter(lambda: fh.read(2), bytes.fromhex("FFD9")):  # iterate until EOI marker
                ba.frombytes(b)
            bits = (*bits, ba)

        img_dec = self.decode(bits, M, N, quality)

        return img_dec.astype(np.uint8)

    def compute_psnr(self, I_dec, I_ref):
        # Input:  an array, I_dec, representing a decoded image in range [0.0,255.0]
        #         an array, I_ref, representing a reference image in range [0.0,255.0]
        # Output: a float, PSNR, representing the PSNR of the decoded image w.r.t. the reference image (in dB)

        # Your code here
        dec_shape = I_dec.shape
        ref_shape = I_ref.shape
        mse = np.sum(
            np.sum(
                np.sum(
                    np.square(
                        I_ref.astype(float) - I_dec.astype(float)),
                    axis=2),
                axis=1),
            axis=0) * (1 / (dec_shape[0] * dec_shape[1]))
        PSNR = 10 * np.log10(255 ** 2 / mse)

        return PSNR
