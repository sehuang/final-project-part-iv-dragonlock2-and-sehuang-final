import os, sys
sys.path.append(os.path.abspath(".."))
from encoder import *
from Helper_functions import *
from PIL import Image
import pathlib

def test_tiff_player():
    curdir = pathlib.Path(os.path.abspath(""))
    path_to_video = curdir.parent / "example_videos" / "dog.png"

def test_image_stack_load():
    curdir = pathlib.Path(os.path.abspath(""))
    path_to_video = str(curdir.parent / "example_videos" / "simple_shape.png")
    imageStack_load(path_to_video)

def test_just_loading_the_png_manually():
    curdir = pathlib.Path(os.path.abspath(""))
    path_to_video = curdir.parent / "example_videos" / "dog.png"
    with Image.open(path_to_video) as img:
        print(img.size)


test_image_stack_load()
# test_just_loading_the_png_manually()