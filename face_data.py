import dataclasses
from PIL import Image, ImageDraw


@dataclasses.dataclass()
class Point:
    x: int
    y: int

    def get_drawing_circle(self, radius=1):
        return [(self.x - radius, self.y - radius), (self.x + radius, self.y + radius)]


@dataclasses.dataclass()
class FaceData:
    """FaceData for facial-keypoints-detection"""

    rowid: int
    png_hash: str
    left_eye_center: Point
    right_eye_center: Point
    left_eye_inner_corner: Point
    left_eye_outer_corner: Point
    right_eye_inner_corner: Point
    right_eye_outer_corner: Point
    left_eyebrow_inner_end: Point
    left_eyebrow_outer_end: Point
    right_eyebrow_inner_end: Point
    right_eyebrow_outer_end: Point
    nose_tip: Point
    mouth_left_corner: Point
    mouth_right_corner: Point
    mouth_center_top_lip: Point
    mouth_center_bottom_lip: Point

    def draw_facepoints_on_image(self, im: Image) -> None:
        """
        Draw all the facepoints on the image.
        """

        draw = ImageDraw.Draw(im)
        if self.left_eye_center.x is not None:
            draw.ellipse(self.left_eye_center.get_drawing_circle(), fill="orange")

        if self.left_eye_inner_corner.x is not None:
            draw.ellipse(self.left_eye_inner_corner.get_drawing_circle(), fill="orange")

        if self.left_eye_outer_corner.x is not None:
            draw.ellipse(self.left_eye_outer_corner.get_drawing_circle(), fill="orange")

        if self.right_eye_center.x is not None:
            draw.ellipse(self.right_eye_center.get_drawing_circle(), fill="orange")

        if self.right_eye_inner_corner.x is not None:
            draw.ellipse(
                self.right_eye_inner_corner.get_drawing_circle(), fill="orange"
            )

        if self.right_eye_outer_corner.x is not None:
            draw.ellipse(
                self.right_eye_outer_corner.get_drawing_circle(), fill="orange"
            )

        if self.left_eyebrow_inner_end.x is not None:
            draw.ellipse(self.left_eyebrow_inner_end.get_drawing_circle(), fill="green")

        if self.left_eyebrow_outer_end.x is not None:
            draw.ellipse(self.left_eyebrow_outer_end.get_drawing_circle(), fill="green")

        if self.right_eyebrow_inner_end.x is not None:
            draw.ellipse(
                self.right_eyebrow_inner_end.get_drawing_circle(), fill="green"
            )

        if self.right_eyebrow_outer_end.x is not None:
            draw.ellipse(
                self.right_eyebrow_outer_end.get_drawing_circle(), fill="green"
            )

        if self.nose_tip.x is not None:
            draw.ellipse(self.nose_tip.get_drawing_circle(), fill="yellow")

        if self.mouth_left_corner.x is not None:
            draw.ellipse(self.mouth_left_corner.get_drawing_circle(), fill="magenta")

        if self.mouth_right_corner.x is not None:
            draw.ellipse(self.mouth_right_corner.get_drawing_circle(), fill="magenta")

        if self.mouth_center_top_lip.x is not None:
            draw.ellipse(self.mouth_center_top_lip.get_drawing_circle(), fill="magenta")

        if self.mouth_center_bottom_lip.x is not None:
            draw.ellipse(
                self.mouth_center_bottom_lip.get_drawing_circle(), fill="magenta"
            )
