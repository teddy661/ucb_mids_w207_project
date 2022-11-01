from io import BytesIO
from math import ceil
from PIL import Image
import os

import db.db_access as dba
from db.create_db import get_paths
from face_data import FaceData

IMAGE_LIMT = 100
IMAGE_PER_ROW = 10


def main():
    TRAIN_DATA, TEST_DATA, TRAIN_DB, TEST_DB = get_paths()
    sqcon, sqcur = dba.get_con_and_cursor(TRAIN_DB)

    #### Missing features
    data_cols = dba.get_data_column_names(sqcur)
    for col in data_cols:
        print(
            f"Feature: {col:<25} Missing: {len(dba.get_images_missing_data(sqcur, col))}"
        )

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

    ##### Counts up duplicate pictures
    duplicate_images = dba.get_duplicate_images_with_count(sqcur)
    for image_row in duplicate_images:
        rows = dba.get_rows_from_hash(sqcur, image_row[1])
        tgt = Image.new("RGB", (96 * len(rows), 96))
        x = 0
        for row in rows:
            face_points = dba.get_face_points_from_db(sqcur, row[0])
            im = face_points.draw_facepoints_on_image(image_row[2])
            tgt.paste(im, (x * 96, 0))
            x = x + 1

        print(f"Saving {image_row[1]}_dups.png")
        if len(rows) > 2:  # only saving images with more than 2 duplicates
            tgt.save(f"{folder}{image_row[1]}_dups.png", format="png", optimize=True)

    ##### Render composite with all images and facepoints
    print("")
    print("")
    print("Annotating All Images...")
    unique_images = dba.get_all_images(sqcur)
    unique_images_annotated = []
    for image_row in unique_images:
        face_points = dba.get_face_points_from_db(sqcur, image_row[0])
        im = face_points.draw_facepoints_on_image(image_row[2])
        unique_images_annotated.append(im)

    num_rows = ceil(IMAGE_LIMT / IMAGE_PER_ROW)
    dst = Image.new("RGB", (96 * IMAGE_PER_ROW, 96 * num_rows))
    i = 0
    print(f"Build composite image for top {IMAGE_LIMT} images...")
    for x in range(IMAGE_PER_ROW):
        for y in range(num_rows):
            dst.paste(unique_images_annotated[i], (x * 96, y * 96))
            i = i + 1

    print("Save composite image...")
    dst.save("composite_image.png", format="png", optimize=True)

    dba.dispose(sqcon, sqcur)


if __name__ == "__main__":
    main()
