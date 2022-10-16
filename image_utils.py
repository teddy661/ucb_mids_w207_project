import argparse
import dataclasses
import re
import sqlite3
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from facedata import FaceData
from facedata import Point


def get_image_from_db(sqcur, rowid):
    """

    :param sqcur:
    :type sqcur:
    :return:
    :rtype:
    """
    sqcur.execute(
        """
        SELECT
            rowid,
            png_hash,
            png_image
        FROM
            image_data
        WHERE
            rowid = ?
        """,
        (rowid,),
    )
    rows = sqcur.fetchall()
    if len(rows) != 1:
        print("get_image_from_db returned more than one row. That's very bad. ")
        return None
    else:
        return rows[0][2]


def get_images_missing_data(sqcur, col_root_name):
    query = """
    SELECT
        rowid,
        png_hash
    FROM
        image_data
    WHERE
        rowid IN (
            SELECT
                rowid
            FROM
                image_data
            WHERE
                {0}_x IS NULL
            UNION
            SELECT
                rowid 
            FROM 
                image_data
            WHERE
                {0}_y IS NULL);
                """.format(
        col_root_name
    )
    sqcur.execute(query)
    rows = sqcur.fetchall()
    return rows


def get_all_images(sqcur):
    sqcur.execute(
        """
    SELECT  
        rowid,png_hash,png_image
    FROM
        image_data
    """
    )
    rows = sqcur.fetchall()
    return rows


def get_duplicate_images_with_count(sqcur):
    sqcur.execute(
        """
    SELECT
        COUNT(png_hash),
        png_hash
    FROM
        dup_images
    GROUP BY
        png_hash
    ORDER BY
        COUNT(png_hash) ASC;
    """
    )
    rows = sqcur.fetchall()
    return rows


def get_data_column_names(sqcur):
    sqcur.execute(
        """
    SELECT
        name
    FROM
        PRAGMA_TABLE_INFO ('image_data');
    """
    )
    rows = sqcur.fetchall()
    valid = re.compile("\w+(_x|_y)$")
    point_cols = []
    for col in rows:
        if valid.fullmatch(col[0]):
            point_cols.append(col[0][:-2])
    return sorted(set(point_cols))


def get_face_points_from_db(sqcur, rowid):
    """

    :param sqcur:
    :type sqcur:
    :return:
    :rtype:
    """
    sqcur.execute(
        """
        SELECT
            rowid,
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
            mouth_center_bottom_lip_y
        FROM
            image_data
        WHERE
            rowid = ?
        """,
        (rowid,),
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


def draw_facepoints_on_image(image_data, face_points):
    im = Image.open(image_data).convert("RGB")
    draw = ImageDraw.Draw(im)
    if face_points.left_eye_center.x is not None:
        draw.point(dataclasses.astuple(face_points.left_eye_center), fill="red")

    if face_points.left_eye_inner_corner.x is not None:
        draw.point(dataclasses.astuple(face_points.left_eye_inner_corner), fill="red")

    if face_points.left_eye_outer_corner.x is not None:
        draw.point(dataclasses.astuple(face_points.left_eye_outer_corner), fill="red")

    if face_points.right_eye_center.x is not None:
        draw.point(dataclasses.astuple(face_points.right_eye_center), fill="orange")

    if face_points.right_eye_inner_corner.x is not None:
        draw.point(
            dataclasses.astuple(face_points.right_eye_inner_corner), fill="orange"
        )

    if face_points.right_eye_outer_corner.x is not None:
        draw.point(
            dataclasses.astuple(face_points.right_eye_outer_corner), fill="orange"
        )

    if face_points.left_eyebrow_inner_end.x is not None:
        draw.point(
            dataclasses.astuple(face_points.left_eyebrow_inner_end), fill="green"
        )

    if face_points.left_eyebrow_outer_end.x is not None:
        draw.point(
            dataclasses.astuple(face_points.left_eyebrow_outer_end), fill="green"
        )

    if face_points.right_eyebrow_inner_end.x is not None:
        draw.point(
            dataclasses.astuple(face_points.right_eyebrow_inner_end), fill="blue"
        )

    if face_points.right_eyebrow_outer_end.x is not None:
        draw.point(
            dataclasses.astuple(face_points.right_eyebrow_outer_end), fill="blue"
        )

    if face_points.nose_tip.x is not None:
        draw.point(dataclasses.astuple(face_points.nose_tip), fill="yellow")

    if face_points.mouth_left_corner.x is not None:
        draw.point(dataclasses.astuple(face_points.mouth_left_corner), fill="magenta")

    if face_points.mouth_right_corner.x is not None:
        draw.point(dataclasses.astuple(face_points.mouth_right_corner), fill="magenta")

    if face_points.mouth_center_top_lip.x is not None:
        draw.point(
            dataclasses.astuple(face_points.mouth_center_top_lip), fill="magenta"
        )

    if face_points.mouth_center_bottom_lip.x is not None:
        draw.point(
            dataclasses.astuple(face_points.mouth_center_bottom_lip), fill="magenta"
        )

    return im


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

    #### Missing features
    data_cols = get_data_column_names(sqcur)
    for col in data_cols:
        print(f"Feature: {col:<25} Missing: {len(get_images_missing_data(sqcur, col))}")

    ##### Counts up duplicate pictures
    print("")
    print("")
    duplicate_images = get_duplicate_images_with_count(sqcur)
    for image in duplicate_images:
        print(f"Count: {image[0]} Hash (first 10 only): {image[1][:10]}")

    ##### Render composite with all images and facepoints
    unique_images = get_all_images(sqcur)
    unique_images_annotated = []
    for image in unique_images:
        face_points = get_face_points_from_db(sqcur, image[0])
        image_data = get_image_from_db(sqcur, image[0])
        buf_image_data = BytesIO(image_data)
        unique_images_annotated.append(
            draw_facepoints_on_image(buf_image_data, face_points)
        )
        # draw = ImageDraw.Draw(image_with_points)
        # draw.text((0,0),"fe5952d06f166324b687e4729aead63b3322919f25daa4510571c0ff25dc6b8b7cdebf9cff5f22968014e2730db59bd364fbe5a96757d4232e6e5a5dc091468b", (0,0,0))
        # image_with_points.show()
    # factors of 7049
    # 1 | 7 | 19 | 53 | 133 | 371 | 1007 | 7049
    dst = Image.new("RGB", (96 * 19, 96 * 371))
    i = 0
    for y in range(371):
        for x in range(19):
            dst.paste(unique_images_annotated[i], (x * 96, y * 96))
            i = i + 1
    dst.show()
    sqcon.close()


if __name__ == "__main__":
    main()
