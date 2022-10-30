import re
import numpy as np
import sqlite3
from pathlib import Path
import PIL.Image as Image

from db.image_loader import create_image_from_pixels
from face_data import FaceData
from face_data import Point


def get_con_and_cursor(db_path: Path):

    if db_path.is_file():
        db = db_path.resolve()
    else:
        print("Database didn't exist! Try again")
        exit()

    sqcon = sqlite3.connect(db)
    sqcur = sqcon.cursor()

    return sqcon, sqcur


def dispose(sqcon: sqlite3.Connection, sqcur: sqlite3.Cursor):
    """Dispose the connection and cursor"""
    sqcur.close()
    sqcon.close()
    del sqcur
    del sqcon


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
        rowid,
        png_hash,
        png_image
    FROM
        image_data
    """
    )
    rows = sqcur.fetchall()
    return rows


def get_training_data_as_numpy(sqcur):
    sqcur.execute(
        """
    SELECT  
        image_raw_pixels,
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
    """
    )
    rows = sqcur.fetchall()

    X = np.stack([np.asarray(create_image_from_pixels(row[0])) for row in rows])
    y = np.asarray([row[1:] for row in rows])

    print(f"Image array shape: {X.shape}, Label array shape: {y.shape}")

    return X, y


def get_test_data_as_numpy(sqcur):
    sqcur.execute(
        """
    SELECT  
        image_raw_pixels
    FROM
        image_data
    """
    )
    rows = sqcur.fetchall()

    X = np.stack([np.asarray(create_image_from_pixels(row[0])) for row in rows])

    print(f"Image array shape: {X.shape}")

    return X


def get_duplicate_images_with_count(sqcur):
    sqcur.execute(
        """
    SELECT
        COUNT(png_hash),
        png_hash,
        png_image
    FROM
        image_data
    GROUP BY
        png_hash,
        png_image
    HAVING
        COUNT(png_hash)>1
    ORDER BY
        COUNT(png_hash) DESC;
    """
    )
    rows = sqcur.fetchall()
    return rows


def get_rowid_from_hash(sqcur, png_hash):
    sqcur.execute(
        """
    SELECT
        rowid
    FROM
        image_data
    WHERE
        png_hash = ?
    ORDER BY 
        rowid ASC;
    """,
        (png_hash,),
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


def get_face_points_from_db(sqcur, rowid) -> FaceData or None:
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

    row = rows[0]
    result = FaceData(
        row[0],
        row[1],
        Point(row[2], row[3]),
        Point(row[4], row[5]),
        Point(row[6], row[7]),
        Point(row[8], row[9]),
        Point(row[10], row[11]),
        Point(row[12], row[13]),
        Point(row[14], row[15]),
        Point(row[16], row[17]),
        Point(row[18], row[19]),
        Point(row[20], row[21]),
        Point(row[22], row[23]),
        Point(row[24], row[25]),
        Point(row[26], row[27]),
        Point(row[28], row[29]),
        Point(row[30], row[31]),
    )
    return result
