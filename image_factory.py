import argparse
import sqlite3
from io import BytesIO
from pathlib import Path
from PIL import Image
import os

from db.db_utils import *


def main():

    check_file = (Path(__file__).parent / "db/training.db").resolve()
    if check_file.is_file():
        db = check_file.resolve()
    else:
        print("Database didn't exist! Try again")
        exit()

    sqcon = sqlite3.connect(db)
    sqcur = sqcon.cursor()

    #### Missing features
    data_cols = get_data_column_names(sqcur)
    for col in data_cols:
        print(f"Feature: {col:<25} Missing: {len(get_images_missing_data(sqcur, col))}")

    ##### Counts up duplicate pictures
    print("")
    print("")
    # delete images that already exist
    folder = "dupes/"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))

    duplicate_images = get_duplicate_images_with_count(sqcur)
    for image in duplicate_images:
        print(f"Count: {image[0]} Hash (first 10 only): {image[1]}")
        rowids = get_rowid_from_hash(sqcur, image[1])
        tgt = Image.new("RGB", (96 * len(rowids), 96))
        x = 0
        for rowid in rowids:
            face_points = get_face_points_from_db(sqcur, rowid[0])
            im = Image.open(BytesIO(image[2])).convert("RGB")
            face_points.draw_facepoints_on_image(im)
            tgt.paste(im, (x * 96, 0))
            x = x + 1

        print(f"Saving {image[1]}_dups.png")
        tgt.save(f"{folder}{image[1]}_dups.png", format="png", optimize=True)

    ##### Render composite with all images and facepoints
    print("")
    print("")
    print("Annotating All Images")
    unique_images = get_all_images(sqcur)
    unique_images_annotated = []
    for image in unique_images:
        face_points: FaceData = get_face_points_from_db(sqcur, image[0])
        image_data = get_image_from_db(sqcur, image[0])
        buf_image_data = Image.open(BytesIO(image_data)).convert("RGB")
        face_points.draw_facepoints_on_image(buf_image_data)

        unique_images_annotated.append(buf_image_data)
    # factors of 7049
    # 1 | 7 | 19 | 53 | 133 | 371 | 1007 | 7049
    dst = Image.new("RGB", (96 * 19, 96 * 371))
    i = 0
    print("Build composite image")
    for y in range(371):
        for x in range(19):
            dst.paste(unique_images_annotated[i], (x * 96, y * 96))
            i = i + 1

    print("Save composite image")
    dst.save("composite_image.png", format="png", optimize=True)

    sqcon.close()


if __name__ == "__main__":
    main()
