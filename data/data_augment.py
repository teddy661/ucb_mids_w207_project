import numpy as np
from scipy import ndimage


def flipping_augmentation(images, features):
    flipped_images = np.flip(images, axis=2)

    flipped_features = features.copy()
    for i, feat in enumerate(flipped_features):
        for j, val in enumerate(feat):
            if j % 2 == 0:
                flipped_features[i][j] = 96 - val

    return flipped_images, flipped_features


def rotate_points(points, angle):
    """
    shift points in the plane so that the center of rotation is at the origin
    Our image is 96*96 ,so substract 48
    """
    points = points - 48

    # rotation matrix
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    # rotate the points
    for i in range(0, len(points), 2):
        xy = np.array([points[i], points[i + 1]])
        xy_rot = R @ xy
        points[i], points[i + 1] = xy_rot

    #  shift again so the origin goes back to the desired center of rotation
    points = points + 48
    return points


def rotate_augmentation(images, features, angle=15):
    rotated_images = []
    for img in images:
        img_rot = ndimage.rotate(img, -angle, reshape=False)
        rotated_images.append(img_rot)

    rotated_features = []
    for feat in features:
        feat_rot = rotate_points(feat, angle)
        rotated_features.append(feat_rot)

    return np.array(rotated_images), np.array(rotated_features)


def brightness_augmentation(images, features, factor=1.5):
    bright = []
    for img in images:
        bright.append(np.clip(img * factor, 0, 255))
    return np.array(bright), features


def noise_augmentation(images, features, factor=0.1):
    augmented = []
    noise = np.random.randint(low=0, high=255, size=images.shape[1:])
    for img in images:
        img = img + (noise * factor)
        augmented.append(img)

    return np.array(augmented), features


def augment_data(X, y):
    """
    Augment data by flipping, rotating, adding noise and changing brightness
    """
    augmented_X = [X]
    augmented_y = [y]
    for aug in [
        # flipping_augmentation,
        # rotate_augmentation,
        # noise_augmentation,
        brightness_augmentation,
    ]:
        aug_images, aug_features = aug(X, y)
        augmented_X.append(aug_images)
        augmented_y.append(aug_features)

    return np.concatenate(augmented_X), np.concatenate(augmented_y)
