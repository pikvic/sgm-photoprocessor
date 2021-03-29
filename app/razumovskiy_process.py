import numpy as np
import imutils
import cv2
import re
import csv
from imutils import perspective
from imutils import contours
from shutil import copyfile
from os import listdir, remove
from os.path import isfile, join
from scipy.spatial import distance as dist
from PIL import Image
from pathlib import Path
import shutil


def get_object_info(num, file):
    with open(file, newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['NOM'].lstrip("0") == num:
                return row
    return None


work_folder = Path("data")
result_folder = work_folder / "output_images"
photo_dir = work_folder / "input_images"
csv_dir = work_folder / "database"
csv_file = csv_dir / "razumovskiy.csv"
tmp_file = work_folder / "tmp.jpg"
files = [file for file in photo_dir.iterdir()]


for file in files:
    if file.name.lower().endswith(".jpg"):
        print(file)
        shutil.copy(file, tmp_file)
        number = re.findall(r"\d+", file.name)[-1].lstrip("0")
        object_info = get_object_info(number, csv_file)
 
        if object_info is not None and object_info['SIZE1'] and object_info['SIZE2']:
            print(number, object_info['SIZE1'], object_info['SIZE2'])
 
            # load our input image, convert it to grayscale, and blur it slightly
            image = cv2.imread(str(tmp_file))
            
            max_size = max(float(object_info['SIZE1'].replace(",", ".")), float(object_info['SIZE1'].replace(",", ".")))
            max_pixels = max(image.shape[0], image.shape[1])

            pixels_in_sm = int(max_pixels / max_size)
            basewidth = pixels_in_sm * 5

            print("pixels: " + str(max_pixels) + "\nsize: " + str(max_size) + "\npixels_in_sm: " + str(
                pixels_in_sm) + "\n")

            ruler = Image.open(work_folder / "ruler.png")
            wpercent = (basewidth / float(ruler.size[0]))
            hsize = int((float(ruler.size[1]) * float(wpercent)))
            ruler = ruler.resize((basewidth, hsize), Image.ANTIALIAS)
            w, h = ruler.size

            background = Image.fromarray(np.uint8(image))
            width, height = background.size

            image_box = (int((width / 2) - (w / 2)), (height - h) - int(height / 100))

            background.paste(ruler, image_box, ruler)
            
            cv2.imwrite(str(result_folder / file.name), np.asarray(background))
        else:
            copyfile(tmp_file, file.name)

        #file = open(file_name + ".txt", "w", encoding='utf-8')
        #file.write("№" + object_info['NOM'] + ", " + object_info['NAZ'])
        #file.close()

    #tmp_file.unlink()