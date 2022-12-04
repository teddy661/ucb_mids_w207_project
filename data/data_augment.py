# OpenCV for s a library containing programming functions
# mainly aimed at real-time computer vision problem solving
import cv2


def aug_shift(X, y, pixel_shifts=[15]):
    """
    Function to shift images (X) and points (y) in the same time.
    INPUT:
        X: numpy array with shape (n, d, d, c)
        y: points to plot with shape (n, m)
        pixel_shifts: a list of values indicating horizontal & vertical shift amount in pixels
    OUTPUT:
        augmented images with shape (n, d, d, c)
        augmented points with shape (n, m)
    """
    size = X.shape[1]
    shifted_images = []
    shifted_keypoints = []
    for shift in pixel_shifts:
        for (shift_x, shift_y) in [
            (-shift, -shift),
            (-shift, shift),
            (shift, -shift),
            (shift, shift),
        ]:
            sh = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

            for image, keypoint in zip(X, y):
                shifted_image = cv2.warpAffine(
                    image, sh, (size, size), flags=cv2.INTER_CUBIC
                )
                shifted_keypoint = np.array(
                    [
                        (point + shift_x) if idx % 2 == 0 else (point + shift_y)
                        for idx, point in enumerate(keypoint)
                    ]
                )

                if np.all(0.0 < shifted_keypoint) and np.all(shifted_keypoint < size):
                    shifted_images.append(shifted_image.reshape(size, size, 1))
                    shifted_keypoints.append(shifted_keypoint)
    shifted_keypoints = np.clip(shifted_keypoints, 0.0, size)
    return np.array(shifted_images), np.array(shifted_keypoints)


def aug_brightness(X, y, brightness_range=[0.6, 1.2]):
    """
    Function to adjust the brightness of images (X)
    INPUT:
        X: numpy array with shape (n, d, d, c)
        y: points to plot with shape (n, m)
        brightness_range: a list of two values to decrease/increase brightness
    OUTPUT:
        augmented images with shape (n, d, d, c)
        augmented points with shape (n, m)
    Note:
        Brightness is pre-defined as either 1.2 times or 0.6 times
    """
    altered_brightness_images = []
    inc_brightness_images = np.clip(X * brightness_range[1], 0, 255)
    dec_brightness_images = np.clip(X * brightness_range[0], 0, 255)
    altered_brightness_images.extend(inc_brightness_images)
    altered_brightness_images.extend(dec_brightness_images)
    return np.array(altered_brightness_images), np.concatenate((y, y))


def aug_rotation(X, y, rotation_angles=[15]):
    """
    Function to rotate images (X) and points (y) in the same time.
    INPUT:
        X: numpy array with shape (n, d, d, c)
        y: points to plot with shape (n, m)
        rotation_angles: a list of angles to rotate
    OUTPUT:
        augmented images with shape (n, d, d, c)
        augmented points with shape (n, m)

    """

    rotated_images = []
    rotated_keypoints = []

    size = X.shape[1]  # suppose h == w

    center = (int(size / 2), int(size / 2))

    for angle in rotation_angles:
        for angle in [angle, -angle]:
            rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            angle_rad = -angle * pi / 180.0

            for image in X:
                rotated_image = cv2.warpAffine(
                    image.reshape(size, size), rot, (size, size), flags=cv2.INTER_CUBIC
                )
                rotated_images.append(rotated_image)

            for keypoint in y:
                rotated_keypoint = keypoint - int(size / 2)

                for idx in range(0, len(rotated_keypoint), 2):
                    rotated_keypoint[idx] = rotated_keypoint[idx] * cos(
                        angle_rad
                    ) - rotated_keypoint[idx + 1] * sin(angle_rad)
                    rotated_keypoint[idx + 1] = rotated_keypoint[idx] * sin(
                        angle_rad
                    ) + rotated_keypoint[idx + 1] * cos(angle_rad)
                rotated_keypoint += size / 2
                rotated_keypoints.append(rotated_keypoint)

    return np.reshape(rotated_images, (-1, size, size, 1)), np.array(rotated_keypoints)
