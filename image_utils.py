import argparse
import sqlite3
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw


def get_image(sqcur):
    sqcur.execute("""SELECT png_image FROM training_data LIMIT 1""")
    rows = sqcur.fetchall()
    image = rows[0][0]
    return image

##
## Some proof of concept image stuff. To how it works.
##
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
        "-f",
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
    image_data = get_image(sqcur)
    buf_image_data = BytesIO(image_data)
    im = Image.open(buf_image_data).convert("RGB")
    draw = ImageDraw.Draw(im)
    sqcur.execute(
        """SELECT left_eye_center_x,left_eye_center_y FROM training_data LIMIT 1"""
    )
    rows = sqcur.fetchall()
    draw.point([rows[0][0], rows[0][1]], fill="red")
    im.show()
    sqcon.close()


if __name__ == "__main__":
    main()
