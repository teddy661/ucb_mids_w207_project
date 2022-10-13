import argparse
import sqlite3
import dataclasses
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
from facedata import FaceData
from facedata import Point


def get_image_from_db(sqcur):
    """

    :param sqcur:
    :type sqcur:
    :return:
    :rtype:
    """
    sqcur.execute("""SELECT rowid, png_hash, png_image FROM image_data LIMIT 1""")
    rows = sqcur.fetchall()
    image = rows[0][2]
    return image


def get_face_points_from_db(sqcur, rowid, png_hash):
    """

    :param sqcur:
    :type sqcur:
    :return:
    :rtype:
    """
    sqcur.execute(
        """SELECT   rowid, 
                    png_hash,   
                    left_eye_center_x, 
                    left_eye_center_y, 
                    right_eye_center_x,
                    right_eye_center_y,
                    left_eye_inner_corner_x,
                    left_eye_inner_corner_y,
                    left_eye_outer_corner_x,
                    left_eye_outer_corner_y,
                    right_eye_inner_corner_x,
                    right_eye_inner_corner_y,
                    right_eye_outer_corner_x,
                    right_eye_outer_corner_y,
                    left_eyebrow_inner_end_x,
                    left_eyebrow_inner_end_y,
                    left_eyebrow_outer_end_x,
                    left_eyebrow_outer_end_y,
                    right_eyebrow_inner_end_x,
                    right_eyebrow_inner_end_y,
                    right_eyebrow_outer_end_x,
                    right_eyebrow_outer_end_y,
                    nose_tip_x,
                    nose_tip_y,
                    mouth_left_corner_x,
                    mouth_left_corner_y,
                    mouth_right_corner_x,
                    mouth_right_corner_y,
                    mouth_center_top_lip_x,
                    mouth_center_top_lip_y,
                    mouth_center_bottom_lip_x,
                    mouth_center_bottom_lip_y FROM image_data WHERE rowid = ? AND png_hash = ?""",
        (
            rowid,
            png_hash,
        ),
    )
    rows = sqcur.fetchall()
    if len(rows) == 0:
        print("Didn't find any rows. Poorly Specified query?")
        return None
    if len(rows) > 1:
        print("This should Never Happen. More than one row")
        return None
    result = FaceData(
        rows[0][0],
        rows[0][1],
        Point(rows[0][2], rows[0][3]),
        Point(rows[0][4], rows[0][5]),
        Point(rows[0][6], rows[0][7]),
        Point(rows[0][8], rows[0][9]),
        Point(rows[0][10], rows[0][11]),
        Point(rows[0][12], rows[0][13]),
        Point(rows[0][14], rows[0][15]),
        Point(rows[0][16], rows[0][17]),
        Point(rows[0][18], rows[0][19]),
        Point(rows[0][20], rows[0][21]),
        Point(rows[0][22], rows[0][23]),
        Point(rows[0][24], rows[0][25]),
        Point(rows[0][26], rows[0][27]),
        Point(rows[0][28], rows[0][29]),
        Point(rows[0][30], rows[0][31]),
    )
    return result


#
# Some proof of concept image stuff. To how it works.
#
def main():
    """
    Generate sqlite database from a pandas table.
    :return:
    :rtype:
    """
    parser = argparse.ArgumentParser(
        description="Utilities For Working with Face Images",
    )
    parser.add_argument(
        "-d",
        dest="db_file",
        type=str,
        required=True,
        metavar="character_string",
        help="path to database",
    )
    args = parser.parse_args()
    prog_name = parser.prog
    check_file = Path(args.db_file)
    if check_file.is_file():
        db = check_file.resolve()
    else:
        print("Database didn't exist! Try again")
        exit()

    sqcon = sqlite3.connect(db)
    sqcur = sqcon.cursor()

    row_id = 1
    my_png_hash = "2dd2b74d17693c1290fde72ede8c08edcfbcc86818933ab65a4d6e87ae83bd1db305cbff4d65690dbdf1753d905aee7e68cbe7b2c2f4d8d879aa2c655f2c4521"
    face_points = get_face_points_from_db(sqcur, row_id, my_png_hash)
    image_data = get_image_from_db(sqcur)
    buf_image_data = BytesIO(image_data)
    im = Image.open(buf_image_data).convert("RGB")
    draw = ImageDraw.Draw(im)
    rows = sqcur.fetchall()
    draw.point(dataclasses.astuple(face_points.left_eye_center), fill="red")
    draw.point(dataclasses.astuple(face_points.left_eye_inner_corner), fill="red")
    draw.point(dataclasses.astuple(face_points.left_eye_outer_corner), fill="red")
    draw.point(dataclasses.astuple(face_points.right_eye_center), fill="orange")
    draw.point(dataclasses.astuple(face_points.right_eye_inner_corner), fill="orange")
    draw.point(dataclasses.astuple(face_points.right_eye_outer_corner), fill="orange")
    draw.point(dataclasses.astuple(face_points.left_eyebrow_inner_end), fill="green")
    draw.point(dataclasses.astuple(face_points.left_eyebrow_outer_end), fill="green")
    draw.point(dataclasses.astuple(face_points.right_eyebrow_inner_end), fill="blue")
    draw.point(dataclasses.astuple(face_points.right_eyebrow_outer_end), fill="blue")
    draw.point(dataclasses.astuple(face_points.nose_tip), fill="yellow")
    draw.point(dataclasses.astuple(face_points.mouth_left_corner), fill="magenta")
    draw.point(dataclasses.astuple(face_points.mouth_right_corner), fill="magenta")
    draw.point(dataclasses.astuple(face_points.mouth_center_top_lip), fill="magenta")
    draw.point(dataclasses.astuple(face_points.mouth_center_bottom_lip), fill="magenta")
    im.show()
    sqcon.close()


if __name__ == "__main__":
    main()
