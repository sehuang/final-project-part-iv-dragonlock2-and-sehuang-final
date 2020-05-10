import os, sys
sys.path.append(os.path.abspath(".."))
from encoder import *
from Helper_functions import *
import pathlib

def test_tiff_player():
    curdir = pathlib.Path(os.path.abspath(""))
    path_to_video = curdir.parent[0] / "example_videos" / "dog.png"

def test_image_stack_load():
    curdir = pathlib.Path(os.path.abspath(""))
    path_to_video = str(curdir.parent / "example_videos" / "dog.png")
    imageStack_load(path_to_video)

test_image_stack_load()