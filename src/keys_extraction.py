from copy import deepcopy

import cv2
import skimage.io
import numpy as np
import os
from matplotlib import image, pyplot as plt
import skimage.io
import json
import math
import inspect
from skimage.feature import match_template
from sklearn.cluster import DBSCAN

from logger import logger

import os
import cv2
import numpy as np

def _save_step(name, img):
    os.makedirs("outputs/key_steps", exist_ok=True)
    if img is None:
        return

    # uint8 safety
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # grayscale
    if len(img.shape) == 2:
        cv2.imwrite(f"outputs/key_steps/{name}.png", img)
        return

    # RGB/RGBA -> BGR for OpenCV saving
    if img.shape[2] == 4:
        out = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f"outputs/key_steps/{name}.png", out)

import json
import os

def _save_roi_json(x, y, w, h, out_path="outputs/piano_roi.json"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    roi = {
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h)
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(roi, f, indent=2)


class Key:
    def __init__(self, key_img, y_ul, x_ul, y_dr, x_dr, name):
        self.y_ul = y_ul
        self.x_ul = x_ul
        self.y_dr = y_dr
        self.x_dr = x_dr

        self.key_img = key_img
        self.name = name

    def coords(self):
        """

        :return: y_ul, x_ul, y_dr, x_dr
        """
        return self.y_ul, self.x_ul, self.y_dr, self.x_dr

    def image(self):
        return self.key_img

    def __str__(self):
        return self.name


class WhiteKey(Key):
    def __init__(self, key_img, y_ul, x_ul, y_dr, x_dr, name):
        super().__init__(key_img, y_ul, x_ul, y_dr, x_dr, name)
        if self.name[0] in "CDFGA":
            self.black_is_next = True
        else:
            self.black_is_next = False


class BlackKey(Key):
    def __init__(self, key_img, y_ul, x_ul, y_dr, x_dr, name):
        super().__init__(key_img, y_ul, x_ul, y_dr, x_dr, name)


def _key_to_dict(k: Key):
    y1, x1, y2, x2 = k.coords()
    return {"y_ul": int(y1), "x_ul": int(x1), "y_dr": int(y2), "x_dr": int(x2), "name": str(k.name)}

def _save_keys_json(keys_dict, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {name: _key_to_dict(k) for name, k in keys_dict.items()}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def _load_keys_json(json_path, key_cls, key_img):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    for name, v in data.items():
        out[name] = key_cls(
            key_img,
            int(v["y_ul"]), int(v["x_ul"]),
            int(v["y_dr"]), int(v["x_dr"]),
            str(v.get("name", name))
        )
    return out


key_names = {
    0: "C",
    1: "D",
    2: "E",
    3: "F",
    4: "G",
    5: "A",
    6: "B",
}


class KeysExtractorThroughLines:
    def __init__(self):
        self.logger = logger

        ref_piano_path = "G:\\POLIMI_ACADEMICS\\First_Year\\1st_Semester\\099993_Image_Analysis_&_Computer_Vision\\Projects_&_Homeworks\\Project\\IACV-Project\\data\\octave\\octava.png"
        ref_piano = skimage.io.imread(ref_piano_path)
        self.ref_piano = cv2.cvtColor(ref_piano, cv2.COLOR_RGB2GRAY)
        self.logger.info("Keys Extractor created")
        self.debug = True
        self.cache_white_json = "data/octave/octava_w.json"
        self.cache_black_json = "data/octave/octava_b.json"
        self.cache_enabled = True 


    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def _find_piano_contour(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = cv2.adaptiveThreshold(image_gray, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

        contours, hierarchy = cv2.findContours(mask,
                                               cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # find the biggest countour (c) by the area
        c = max(contours, key=cv2.contourArea)

        return image_gray, c

    def _draw_contour(self, image, c):
        x, y, w, h = cv2.boundingRect(c)

        # draw the biggest contour (c) in green
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return x, y, w, h

    def _create_piano_masked(self, image, c):
        mask = np.zeros_like(image)
        x, y, w, h = cv2.boundingRect(c)

        # draw the biggest contour (c) in green
        mask = cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        image[mask == 0] = 0

        return image

    def _find_contour_angle(self, image, c):
        center, hw, angle = cv2.minAreaRect(c)
        h, w = hw

        if w > h:
            angle = 90 + angle

        self.logger.debug(f"Found h of contour is {np.round(h, 2)} px, width is {np.round(w, 2)} px")
        self.logger.debug(f"Found angle of contour is {np.round(angle, 2)} degrees")

        return angle
        # box = cv2.boxPoints(rect)

    def _find_kp(self, image):
        sift = cv2.SIFT.create()
        kp, des = sift.detectAndCompute(image, None)
        return kp, des

    def build_homography(self, img1, kp1, des1, img2, kp2, des2):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good.append(m)

        if len(good) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(img2, [np.int32(dst)],
                                 True, 255, 3, cv2.LINE_AA)

        else:
            self.logger.debug("Not enough matches are found - {}/{}".format(len(good), 10))
            matchesMask = None

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

        plt.imshow(img3, 'gray'), plt.show()

    def _calculate_similarity(self, frame1, frame2):
        h1, w1 = frame1.shape
        h2, w2 = frame2.shape

        h_min = min(h1, h2)
        w_min = min(w1, w2)

        frame1 = deepcopy(frame1[:h_min, :w_min, ...])
        frame2 = deepcopy(frame2[:h_min, :w_min, ...])

        frame1[frame1 > 127] = 255
        frame1[frame1 <= 127] = 0

        frame2[frame2 > 127] = 255
        frame2[frame2 <= 127] = 0

        black1 = np.sum(frame1 == 0)
        black2 = np.sum(frame2 == 0)

        white1 = np.sum(frame1 == 255)
        white2 = np.sum(frame2 == 255)

        eps = 1e-9
        score = (black1 / (white1 + eps)) / (black2 / (white2 + eps) + eps)

        if score > 1:
            score = 1 / score

        return score
    
    
    def _make_white_key_obj(self, key_img, x1, y1, x2, y2, name):
        # WhiteKey(key_img, y_ul, x_ul, y_dr, x_dr, name)
        return WhiteKey(
            key_img,
            int(y1), int(x1),
            int(y2), int(x2),
            str(name)
        )


    def _build_white_keys_grid(self, x, y, w, h, white_key_w, white_key_h, masked_piano):
        """
        Stable white keys are generated as a uniform grid.
        This avoids border noise problems.
        Coordinates are in FULL IMAGE system (x,y are ROI offsets).
        """

        # safety
        white_key_w = int(max(6, round(float(white_key_w))))
        white_key_h = int(max(20, round(float(white_key_h))))

        # limits inside ROI
        x_start = int(x + 2)
        x_end = int(x + w - 2)

        # clamp height inside image
        img_h = masked_piano.shape[0]
        y1 = int(max(0, y))
        y2 = int(min(img_h - 1, y + white_key_h))

        white_keys = {}
        notes = ["C","D","E","F","G","A","B"]

        xi = x_start
        idx = 0

        while xi + white_key_w < x_end:
            name = f"{notes[idx % 7]}_{idx}"

            k = WhiteKey(
                masked_piano,
                int(y1), int(xi),
                int(y2), int(xi + white_key_w),
                name
            )

            white_keys[name] = k

            xi += white_key_w
            idx += 1


        return white_keys


    def _find_white_keys_coords(self, masked_piano, x, y, w, h, white_key_w):
        masked_piano = masked_piano.astype(np.int8)

        keys_dict = {}
        patch = masked_piano[y:y + h, x:x + w, ...]

        # --- HARD FIX for adaptiveThreshold input ---
        print("PATCH BEFORE:", patch.shape, patch.dtype)

        # if patch is RGBA/RGB/BGR, convert to gray
        if len(patch.shape) == 3:
            if patch.shape[2] == 4:
                patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2GRAY)
            else:
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

        # force uint8
        if patch.dtype != np.uint8:
            # if patch is float in [0,1], scale to [0,255]
            if patch.dtype in [np.float32, np.float64] and patch.max() <= 1.0:
                patch = (patch * 255.0).astype(np.uint8)
            else:
                patch = np.clip(patch, 0, 255).astype(np.uint8)

        # make it contiguous (OpenCV sometimes fails without this)
        patch = np.ascontiguousarray(patch)

        print("PATCH AFTER:", patch.shape, patch.dtype)

        patch = cv2.adaptiveThreshold(patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 199, 5)
        
        # ensure patch is grayscale uint8 for adaptiveThreshold
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)  # or BGR2GRAY if patch is BGR
        if patch.dtype != np.uint8:
            patch = np.clip(patch, 0, 255).astype(np.uint8)

        ref_piano_h, ref_piano_w = self.ref_piano.shape[:2]
        ref_piano_new_w = int(white_key_w * 7)
        ref_piano_new_h = int(ref_piano_h * (ref_piano_new_w / ref_piano_w))
        ref_piano_resh = cv2.resize(self.ref_piano, (ref_piano_new_w, ref_piano_new_h))

        result = match_template(patch, ref_piano_resh)


        res_max = np.max(result)

        mask = np.zeros_like(result)
        mask[result > res_max * 0.85] = 1


        peaks_coords_y, peaks_coords_x = np.where(mask == 1)

        peaks_coords = np.array(list(zip(peaks_coords_y, peaks_coords_x)))
        clustering_octave = DBSCAN(eps=3, min_samples=2).fit(peaks_coords)
        labels_octave = clustering_octave.labels_

        # left upper coordinates of the "do(C)" key in each found octave
        octave_coords_lu = []
        for label in np.unique(labels_octave):
            coord_y = int(np.mean(peaks_coords_y[labels_octave == label])) + y
            coord_x = int(np.mean(peaks_coords_x[labels_octave == label])) + x

            for j in range(5):
                if np.min(masked_piano[coord_y, coord_x - 5:coord_x + 5]) - np.min(
                        masked_piano[coord_y - j, coord_x - 5:coord_x + 5]) > 150:
                    coord_y = coord_y - j
                    break
            octave_coords_lu.append([coord_y, coord_x])

        octave_coords_lu.sort(key=lambda x: x[1])
        diff = [octave_coords_lu[i + 1][1] - octave_coords_lu[i][1] for i in range(len(octave_coords_lu) - 1)]
        diff.append(diff[-1])

        octave_coords_lu = np.array(octave_coords_lu)
        img_h, img_w = masked_piano.shape[:2]

        y_coord_bottom_keys = []
        for coord in octave_coords_lu:
            y, x = coord
            for j in range(y, img_h):
                if np.mean(masked_piano[j:j + 3, x - 5:x + 5]) / np.mean(masked_piano[j + 3:j + 6, x - 5:x + 5]) > 2:
                    y_coord_bottom_keys.append(j + 3)
                    break
        h_keys = int(np.median(y_coord_bottom_keys) - np.median(octave_coords_lu[:, 0]))

        self.logger.info(f"Found height of white keys is {h_keys}")

        all_white_keys_coords = []
        num_octave = 1
        # add coordinates of keys in found octaves
        for idx, coord in enumerate(octave_coords_lu):
            y, x = coord
            w_keys = diff[idx] / 7

            for i in range(7):
                name_cur = f"{key_names[i]}_{num_octave}"
                key_cur = WhiteKey(masked_piano[y: y + h_keys, int(x + i * w_keys):int(x + (i + 1) * w_keys)],
                                   y, int(x + i * w_keys), y + h_keys, int(x + (i + 1) * w_keys), name_cur)
                keys_dict[name_cur] = key_cur
                all_white_keys_coords.append([y, int(x + i * w_keys), y + h_keys, int(x + (i + 1) * w_keys)])

            num_octave += 1

        # add coordinates of keys not in full octave,
        # going before the first octave
        start_y, start_x, _, _ = keys_dict["C_1"].coords()
        for j in range(14):
            if int(start_x - white_key_w * (j % 7 + 1)) + 5 < 0:
                break

            patch_key = masked_piano[start_y: start_y + h_keys - 1,
                        int(start_x - white_key_w * (j % 7 + 1)): int(start_x - white_key_w * (j % 7))]
            name_key = key_names[6 - j % 7]
            true_key = keys_dict[f"{name_key}_2"].image()

            score = self._calculate_similarity(patch_key, true_key)

            if score < 0.5:
                break

            name_cur = f"{name_key}_0"
            key_cur = WhiteKey(patch_key, start_y, int(start_x - white_key_w * (j % 7 + 1)),
                               start_y + h_keys - 1,
                               int(start_x - white_key_w * (j % 7)),
                               name_cur)
            keys_dict[name_cur] = key_cur

        # add coordinates of keys not in full octave,
        # going after the last octave
        start_y, _, _, start_x = keys_dict[f"B_{num_octave - 1}"].coords()
        for j in range(14):
            if int(start_x + white_key_w * (j % 7 + 1)) - 5 > img_w:
                break

            patch_key = masked_piano[start_y: start_y + h_keys - 1,
                        int(start_x + white_key_w * (j % 7)): int(start_x + white_key_w * ((j % 7) + 1))]

            name_key = key_names[j]
            true_key = keys_dict[f"{name_key}_2"].image()

            score = self._calculate_similarity(patch_key, true_key)
            if score < 0.5:
                # it can be the last "C" which looks different from usual "C"
                if np.sum(patch_key > 200) / (patch_key.shape[0] * patch_key.shape[1]) < 0.8:
                    break

            name_cur = f"{name_key}_{num_octave}"
            key_cur = WhiteKey(patch_key, start_y, int(start_x + white_key_w * (j % 7)),
                               start_y + h_keys - 1,
                               int(start_x + white_key_w * (j % 7 + 1)),
                               name_cur)
            keys_dict[name_cur] = key_cur

        return keys_dict, h_keys

    def _find_black_keys_w(self, masked_piano, x, y, w, h, white_key_h, white_keys_coords):
        patch = masked_piano[y:y + h, x:x + w, ...]
        patch = cv2.adaptiveThreshold(patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 199, 5)

        # use first white key as reference instead of named key
        first_key = list(white_keys_coords.values())[0]
        black_key_y = int(first_key.coords()[0] - y + white_key_h / 3)

        black_w_list = []

        cur_len = 0
        for j in range(w):
            if patch[black_key_y, j] == 0:
                cur_len += 1
            else:
                if cur_len > 0:
                    black_w_list.append(cur_len)
                cur_len = 0

        black_w = np.median(black_w_list)

        return black_w

    def _find_black_keys_coords(self, masked_piano, black_w, white_keys_coords):

        # using the info about the location of black keys (between white),
        # derive their coordinates

        white_keys = list(white_keys_coords.keys())
        black_keys = {}
        white_keys.sort(key=lambda x: 100 * int(x[-1]) + "CDEFGAB".index(x[0]))

        masked_piano[masked_piano > 150] = 255
        masked_piano[masked_piano < 150] = 0

        for key_name in white_keys:
            white_key = white_keys_coords[key_name]

            if white_key.black_is_next:
                name_cur = white_key.name
                name_cur = name_cur[0] + "#" + name_cur[1:]

                y_ul, _, y_dr, x_dr = white_key.coords()
                y_ul_b = y_ul
                x_ul_b = int(x_dr - black_w // 2)

                x_dr_b = int(x_ul_b + black_w)
                x_mean_b = int((x_dr_b + x_ul_b) / 2)

                y_dr_b = 0
                for i in range(int((y_dr - y_ul) / 3), y_dr - y_ul):

                    if abs(np.mean(masked_piano[y_ul + i, x_mean_b - 2:x_mean_b + 2]) - np.mean(
                            masked_piano[y_ul + i + 1, x_mean_b - 2:x_mean_b + 2])) > 20:
                        y_dr_b = y_ul + i
                        break

                max_dif = 15

                for i in range(-int(black_w / 4), int(black_w / 4)):

                    cur_dif = abs(np.mean(masked_piano[y_ul + 5:y_ul + 10, x_ul_b + i - 1: x_ul_b + i]) - np.mean(
                        masked_piano[y_ul + 5:y_ul + 10, x_ul_b + i: x_ul_b + i + 1]))

                    if cur_dif > max_dif:
                        max_dif = cur_dif
                        x_ul_b = x_ul_b + i

                max_dif = 15
                for i in range(-int(black_w / 4), int(black_w / 4)):

                    cur_dif = abs(np.mean(masked_piano[y_ul + 5:y_ul + 10, x_dr_b + i - 1: x_dr_b + i]) - np.mean(
                        masked_piano[y_ul + 5:y_ul + 10, x_dr_b + i: x_dr_b + i + 1]))

                    if cur_dif > max_dif:
                        max_dif = cur_dif
                        x_dr_b = x_dr_b + i
                if y_dr_b > 0:
                    black_key = BlackKey(masked_piano[y_ul_b: y_dr_b, x_ul_b:x_dr_b],
                                         y_ul_b, x_ul_b, y_dr_b, x_dr_b, name_cur)
                    black_keys[name_cur] = black_key


        return black_keys

    def _find_white_keys_w(self, image):
        """
        image: grayscale ROI (uint8)
        returns: (image, white_key_w)
        """

        # 1) smooth
        gray = cv2.GaussianBlur(image, (5, 5), 0)

        # 2) vertical edges only (Sobel x)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gx = np.abs(gx)

        # 3) use only the key area (ignore top border reflections)
        h, w = gx.shape
        y1 = int(0.15 * h)
        y2 = int(0.95 * h)
        gx_roi = gx[y1:y2, :]

        # 4) vertical projection (sum of vertical edge strength per column)
        proj = gx_roi.sum(axis=0)

        # 5) normalize + smooth 1D signal
        proj = proj / (proj.max() + 1e-9)
        proj = cv2.GaussianBlur(proj.reshape(1, -1), (1, 31), 0).ravel()

        # 6) find peaks (simple threshold + local max)
        thr = 0.35  # loosen if too few peaks: 0.25
        peaks = []
        for x in range(2, w - 2):
            if proj[x] > thr and proj[x] >= proj[x - 1] and proj[x] >= proj[x + 1]:
                peaks.append(x)

        if len(peaks) < 10:
            raise ValueError("Too few vertical borders found from projection. Increase contrast or lower thr.")

        # 7) remove near-duplicates
        filtered = [peaks[0]]
        for x in peaks[1:]:
            if x - filtered[-1] >= 4:
                filtered.append(x)
        filtered = np.array(filtered)

        # 8) estimate typical spacing (median of differences)
        diffs = np.diff(filtered)
        diffs = diffs[(diffs >= 4) & (diffs <= 80)]  # plausible spacing in ROI
        if len(diffs) == 0:
            raise ValueError("No plausible border spacings from projection.")

        white_key_w = float(np.median(diffs))

        self.logger.info(f"The width of white key is {white_key_w}")
        return image, white_key_w


    def _find_orientation(self, image, y, h):
        mask = cv2.adaptiveThreshold(image, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
        _save_step("04_white_width_mask", mask)

        rows_max = np.mean(mask, axis=1)
        rows_piano = rows_max[y:y + h]
        first_part_mean = np.mean(rows_piano[:len(rows_piano) // 2])
        second_part_mean = np.mean(rows_piano[len(rows_piano) // 2:])

        if second_part_mean < first_part_mean:
            return False

        return True

    def __call__(self, image: np.ndarray):
        """
        this method will return coordinates of white
        and black keys in the current image
        (the image should be without hands!) """

        if self.debug:
            # simulate rotated image
            image = self.rotate_image(image, 0)

        to_draw_img = deepcopy(image)
        _save_step("01_full_input", image)

        # find the initial piano contour
        image_gray, c = self._find_piano_contour(image)

        # define the rotation angle of the contour
        angle = self._find_contour_angle(image_gray, c)

        # rotate image on this angle
        image = self.rotate_image(image, angle)

        to_draw_img = self.rotate_image(to_draw_img, angle)

        # find the new (angle is 0) piano contour
        image_gray, c = self._find_piano_contour(image)

        # draw piano contour on image
        x, y, w, h = self._draw_contour(image, c)
        _save_step("02_roi_box", image)

        # delete irrelevant info from image
        masked_piano = self._create_piano_masked(image_gray, c)
        _save_step("03_masked_piano", masked_piano)

        # if the piano is upside-down, it should be rotated by 180 degrees
        is_correct_orientation = self._find_orientation(masked_piano, y, h)

        if not is_correct_orientation:
            masked_piano = self.rotate_image(masked_piano, 180)
            to_draw_img = self.rotate_image(to_draw_img, 180)
            angle += 180

        # find width of the white key
        masked_piano, white_key_w = self._find_white_keys_w(masked_piano)

        img_h, img_w = masked_piano.shape[:2]

        if white_key_w is None or (isinstance(white_key_w, float) and math.isnan(white_key_w)):
            # assume ~20–30 white keys visible in the frame
            estimated_white_keys = 52
            white_key_w = img_w / estimated_white_keys
            self.logger.warning(f"white_key_w was NaN. Fallback used: {white_key_w}")

        # find coordinates of white keys
        # white_keys_coords, white_key_h = self._find_white_keys_coords(masked_piano, x, y, w, h, white_key_w)
        # self._draw_keys_coords(white_keys_coords, to_draw_img)

        # stable height for white keys (use ROI height)
        white_key_h = int(0.95 * h)

        # build uniform white-key grid in full-image coordinates
        white_keys_coords = self._build_white_keys_grid(x, y, w, h, white_key_w, white_key_h, masked_piano)

        # --- FIX: force reasonable white-key widths ---
        w_target = int(round(white_key_w))
        for k in white_keys_coords.values():
            # coords() returns (y1,x1,y2,x2) in your code style
            y1, x1, y2, x2 = k.coords()

            # keep left border, force width
            x2_new = x1 + w_target

            # prevent overflow outside image
            x2_new = min(x2_new, masked_piano.shape[1] - 1)

            # write back (adjust attribute names if different)
            k.x2 = x2_new

        # masked_piano is grayscale ROI with background removed
        # to_draw_img is rotated full image used for overlays

        if self.cache_enabled and os.path.exists(self.cache_white_json) and os.path.exists(self.cache_black_json):
            # Cached mode: JSON is loaded and drawn on full image
            white_keys_coords = _load_keys_json(self.cache_white_json, WhiteKey, masked_piano)
            black_keys_coords = _load_keys_json(self.cache_black_json, BlackKey, masked_piano)
            angle_out = angle
        else:
            # Normal mode: detection is performed
            masked_piano, white_key_w = self._find_white_keys_w(masked_piano)
            white_key_h = int(0.95 * h)

            white_keys_coords = self._build_white_keys_grid(x, y, w, h, white_key_w, white_key_h, masked_piano)
            black_key_w = self._find_black_keys_w(masked_piano, x, y, w, h, white_key_h, white_keys_coords)
            black_keys_coords = self._find_black_keys_coords(masked_piano, black_key_w, white_keys_coords)

            # Save JSON outputs from detector
            _save_keys_json(white_keys_coords, self.cache_white_json)
            _save_keys_json(black_keys_coords, self.cache_black_json)
            angle_out = angle

        # ---- overlays (always produced) ----
        white_vis = deepcopy(to_draw_img)
        if len(white_vis.shape) == 2:
            white_vis = cv2.cvtColor(white_vis, cv2.COLOR_GRAY2RGB)
        for k in white_keys_coords.values():
            y1, x1, y2, x2 = k.coords()
            cv2.rectangle(white_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _save_step("05_white_keys_only", white_vis)

        black_vis = deepcopy(to_draw_img)
        if len(black_vis.shape) == 2:
            black_vis = cv2.cvtColor(black_vis, cv2.COLOR_GRAY2RGB)
        for k in black_keys_coords.values():
            y1, x1, y2, x2 = k.coords()
            cv2.rectangle(black_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        _save_step("06_black_keys_only", black_vis)

        final_vis = deepcopy(white_vis)
        for k in black_keys_coords.values():
            y1, x1, y2, x2 = k.coords()
            cv2.rectangle(final_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        _save_step("07_final_white_black", final_vis)

        x, y, w, h = self._draw_contour(image, c)
        _save_roi_json(x, y, w, h)


        return white_keys_coords, black_keys_coords, angle_out



def main():
    """
    test KeysExtractorThroughLines and other extractors here
    :return:
    """
    keys_extractor = KeysExtractorThroughLines()
    img_path = "G:\\POLIMI_ACADEMICS\\First_Year\\1st_Semester\\099993_Image_Analysis_&_Computer_Vision\\Projects_&_Homeworks\\Project\\IACV-Project\\data\\frames\\piano_video_1\\cleaned\\cleaned_frame_2.png"
    image = skimage.io.imread(img_path)

    keys_extractor(image)


if __name__ == "__main__":
    main()