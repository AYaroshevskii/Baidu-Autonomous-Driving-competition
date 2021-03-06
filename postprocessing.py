""" 
Image postprocessing and visualization functions.
I find it useful and took it from public kernels
Lets upvote : https://www.kaggle.com/hocop1/centernet-baseline
"""

import numpy as np
import cv2
from scipy.optimize import minimize
from math import sin, cos

DISTANCE_THRESH_CLEAR = 2
IMG_SHAPE = (2710, 3384, 3)

IMG_WIDTH = 1600
IMG_HEIGHT = 704
MODEL_SCALE = 8

camera_matrix = np.array(
    [[2304.5479, 0, 1686.2379], [0, 2305.8757, 1354.9849], [0, 0, 1]], dtype=np.float32
)
camera_matrix_inv = np.linalg.inv(camera_matrix)


def imread(path, fast_mode=False):
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img


def convert_3d_to_2d(x, y, z, fx=2304.5479, fy=2305.8757, cx=1686.2379, cy=1354.9849):
    # stolen from https://www.kaggle.com/theshockwaverider/eda-visualization-baseline
    return x * fx / z + cx, y * fy / z + cy


def optimize_xy(r, c, x0, y0, z0):
    def distance_fn(xyz):
        x, y, z = xyz
        x, y = convert_3d_to_2d(x, y, z0)
        y, x = x, y
        x = (x - IMG_SHAPE[0] // 2) * IMG_HEIGHT / (IMG_SHAPE[0] // 2) / MODEL_SCALE
        x = np.round(x).astype("int")
        y = (y + IMG_SHAPE[1] // 4) * IMG_WIDTH / (IMG_SHAPE[1] * 1.5) / MODEL_SCALE
        y = np.round(y).astype("int")
        return (x - r) ** 2 + (y - c) ** 2

    res = minimize(distance_fn, [x0, y0, z0], method="Powell")
    x_new, y_new, z_new = res.x
    return x_new, y_new, z0


def clear_duplicates(coords):
    for c1 in coords:
        xyz1 = np.array([c1["x"], c1["y"], c1["z"]])
        for c2 in coords:
            xyz2 = np.array([c2["x"], c2["y"], c2["z"]])
            distance = np.sqrt(((xyz1 - xyz2) ** 2).sum())
            if distance < DISTANCE_THRESH_CLEAR:
                if c1["confidence"] < c2["confidence"]:
                    c1["confidence"] = -1
    return [c for c in coords if c["confidence"] > 0]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _regr_back(regr_dict):
    for name in ["x", "y", "z"]:
        regr_dict[name] = regr_dict[name] * 100
    regr_dict["roll"] = rotate(regr_dict["roll"], -np.pi)

    pitch_sin = regr_dict["pitch_sin"] / np.sqrt(
        regr_dict["pitch_sin"] ** 2 + regr_dict["pitch_cos"] ** 2
    )
    pitch_cos = regr_dict["pitch_cos"] / np.sqrt(
        regr_dict["pitch_sin"] ** 2 + regr_dict["pitch_cos"] ** 2
    )
    regr_dict["pitch"] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict


def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)], [0, 1, 0], [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0], [0, cos(pitch), -sin(pitch)], [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0], [sin(roll), cos(roll), 0], [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))


def rotate(x, angle):
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x


def extract_coords(prediction, mode="val", thr=0.3):
    logits = prediction[0]
    regr_output = prediction[1:]

    if mode == "val":
        points = np.argwhere(logits > 0)
    else:
        points = np.argwhere(logits > thr)

    col_names = sorted(["x", "y", "z", "yaw", "pitch_sin", "pitch_cos", "roll"])
    coords = []
    for r, c in points:
        regr_dict = dict(zip(col_names, regr_output[:, r, c]))
        coords.append(_regr_back(regr_dict))
        coords[-1]["confidence"] = logits[r, c]
        coords[-1]["x"], coords[-1]["y"], coords[-1]["z"] = optimize_xy(
            r, c, coords[-1]["x"], coords[-1]["y"], coords[-1]["z"]
        )
    coords = clear_duplicates(coords)
    return coords


def coords2str(coords, names=["yaw", "pitch", "roll", "x", "y", "z", "confidence"]):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return " ".join(s)


# Visualize
def visualize(img, coords):
    # You will also need functions from the previous cells
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31

    img = img.copy()
    for point in coords:
        # Get values
        x, y, z = point["x"], point["y"], point["z"]
        yaw, pitch, roll = -point["pitch"], -point["yaw"], -point["roll"]
        # Math
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.array(
            [
                [x_l, -y_l, -z_l, 1],
                [x_l, -y_l, z_l, 1],
                [-x_l, -y_l, z_l, 1],
                [-x_l, -y_l, -z_l, 1],
                [0, 0, 0, 1],
            ]
        ).T
        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        # Drawing
        img = draw_line(img, img_cor_points)
        img = draw_points(img, img_cor_points[-1:])

    return img


def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return image


def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)
    #         if p_x > image.shape[1] or p_y > image.shape[0]:
    #             print('Point', p_x, p_y, 'is out of image with shape', image.shape)
    return image
