# -*- coding: utf-8 -*-
# USAGE
# python photo_processor.py

# import the necessary packages
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


def list_dir(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


def get_image_id(name):
    return re.findall('\d+', name)


def get_file_prefix(old_prefix):
    if old_prefix == "БП":
        return "Invertebrate"
    elif old_prefix == "ФЛ":
        return "Flora"
    elif old_prefix == "МН":
        return "Mineral"
    elif old_prefix == "ПВ":
        return "Vertebrate"
    elif old_prefix == "ГР":
        return "Ore"
    elif old_prefix == "ИЛ":
        return "StoneProduct"

    return ""


def get_data_filename(col_prefix):
    if col_prefix == "БП":
        return "fbesp.csv"
    elif col_prefix == "ФЛ":
        return "fflor.csv"
    elif col_prefix == "МН":
        return "fmin.csv"
    elif col_prefix == "ПВ":
        return "fposv.csv"
    elif col_prefix == "ГР":
        return "fgr.csv"
    elif col_prefix == "ИЛ":
        return "fizd.csv"

    return ""


def find_max_contour(contours):
    areas = list(map(cv2.contourArea, contours))
    max_area = max(areas)
    max_index = areas.index(max_area)

    return contours[max_index]


def get_object_info(num, file):
    with open(file, newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['NOM'].lstrip("0") == num:
                return row
    return None


def get_object_size(row):
    obj_size = {"size1": 0, "size2": 0}

    row_size1 = row['SIZE1']
    row_size2 = row['SIZE2']
    if row_size1 and row_size2:
        obj_size["size1"] = float(row_size1.replace(",", "."))
        obj_size["size2"] = float(row_size2.replace(",", "."))
        return obj_size
    return obj_size


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


work_folder = Path("data")
result_folder = work_folder / "output_images"
photo_dir = work_folder / "input_images"
csv_dir = work_folder / "database"
tmp_file = work_folder / "tmp.jpg"
files = [file for file in photo_dir.iterdir()]

for file in files:
    print(file)
    if file.lower().endswith(".jpg"):
        copyfile(photo_dir / file, tmp_file)
        number_list = get_image_id(file)

        col_prefix = file[:2]
        csv_filename = get_data_filename(col_prefix)

        if number_list:
            num = number_list[0].lstrip("0")

            object_info = get_object_info(num, csv_dir + csv_filename)

            if object_info is not None:
                result_file_prefix = result_folder + get_file_prefix(col_prefix) + "_" + num

                if object_info['SIZE1'] and object_info['SIZE2']:
                    print(num, object_info['SIZE1'], object_info['SIZE2'])
                    # load our input image, convert it to grayscale, and blur it slightly
                    image = cv2.imread(tmp_file)

                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # gray = cv2.GaussianBlur(gray, (7, 7), 0)
                    gray = cv2.bilateralFilter(gray, -1, 5, 5)

                    # perform edge detection, then perform a dilation + erosion to
                    # close gaps in between object edges
                    edged = cv2.Canny(gray, 50, 100)
                    edged = cv2.dilate(edged, None, iterations=1)
                    edged = cv2.erode(edged, None, iterations=1)

                    # find contours in the edge map
                    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
                    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

                    # sort the contours from left-to-right and initialize the
                    # 'pixels per metric' calibration variable
                    (cnts, _) = contours.sort_contours(cnts)
                    pixelsPerMetric = None

                    max_contour = find_max_contour(cnts)

                    # compute the rotated bounding box of the contour
                    orig = image.copy()
                    box = cv2.minAreaRect(max_contour)
                    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                    box = np.array(box, dtype="int")

                    # order the points in the contour such that they appear
                    # in top-left, top-right, bottom-right, and bottom-left
                    # order, then draw the outline of the rotated bounding
                    # box
                    box = perspective.order_points(box)
                    #cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

                    # loop over the original points and draw them
                    # for (x, y) in box:
                    # cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

                    # unpack the ordered bounding box, then compute the midpoint
                    # between the top-left and top-right coordinates, followed by
                    # the midpoint between bottom-left and bottom-right coordinates
                    (tl, tr, br, bl) = box
                    (tltrX, tltrY) = midpoint(tl, tr)
                    (blbrX, blbrY) = midpoint(bl, br)

                    # compute the midpoint between the top-left and top-right points,
                    # followed by the midpoint between the top-righ and bottom-right
                    (tlblX, tlblY) = midpoint(tl, bl)
                    (trbrX, trbrY) = midpoint(tr, br)

                    # compute the Euclidean distance between the midpoints
                    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                    max_size = max(float(object_info['SIZE1'].replace(",", ".")), float(object_info['SIZE1'].replace(",", ".")))
                    max_pixels = max(dA, dB)

                    pixels_in_sm = int(max_pixels / max_size)
                    basewidth = pixels_in_sm * 5

                    print("pixels: " + str(max_pixels) + "\nsize: " + str(max_size) + "\npixels_in_sm: " + str(
                        pixels_in_sm) + "\n")

                    ruler = Image.open(work_folder + "ruler.png")
                    wpercent = (basewidth / float(ruler.size[0]))
                    hsize = int((float(ruler.size[1]) * float(wpercent)))
                    ruler = ruler.resize((basewidth, hsize), Image.ANTIALIAS)
                    w, h = ruler.size

                    background = Image.fromarray(np.uint8(orig))
                    width, height = background.size

                    image_box = (int((width / 2) - (w / 2)), (height - h) - int(height / 100))

                    background.paste(ruler, image_box, ruler)
                    # orig.show()

                    cv2.imwrite(result_file_prefix + ".jpg", np.asarray(background))
                else:
                    copyfile(tmp_file, result_file_prefix + ".jpg")

                file = open(result_file_prefix + ".txt", "w", encoding='utf-8')
                file.write("№" + object_info['NOM'] + ", " + object_info['NAZ'])
                file.close()

        remove(tmp_file)
