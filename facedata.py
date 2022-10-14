from dataclasses import dataclass


@dataclass()
class Point:
    x: int
    y: int


@dataclass()
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
