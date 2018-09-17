# coding: utf-8


from __future__ import absolute_import, print_function, division
import cv2
import matplotlib.pyplot as plt
import numpy as np

import os


def attach_image(seal_img, img1):
    rows, cols, _ = seal_img.shape
    roi = img1[200:200 + rows, 200:200 + cols]
    seal_gray_img = cv2.cvtColor(seal_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(seal_gray_img, 254, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(seal_img, seal_img, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    img1[200:200 + rows, 200:200 + cols] = dst
    return img1


# create surf object
surf = cv2.xfeatures2d.SURF_create(400)

MIN_MATCH_COUNT = 10


# image align
def align_image(img_origin, img_changed):
    img_kp1, img_des1 = surf.detectAndCompute(img_origin, None)
    img_kp2, img_des2 = surf.detectAndCompute(img_changed, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(img_des1, img_des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        dst_pts = np.float32([img_kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        src_pts = np.float32([img_kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w, d = img_changed.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst_polygon = cv2.perspectiveTransform(pts, M)

        h_o, w_o, _ = img_origin.shape
        img_changed_per_tran = cv2.warpPerspective(img_changed, M, (w_o, h_o))

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
    return img_changed_per_tran, dst_polygon


# calculate the difference
def diff_image(img_origin, image_aligned, dst_polygon):
    img_diff = cv2.subtract(img_origin, image_aligned)

    # scratch wrap pixel
    h, w, _ = image_aligned.shape
    mask_inv = [[False if cv2.pointPolygonTest(np.squeeze(dst_polygon), (x, y), False) > 0 else True for x in range(w)]
                for y in range(h)]
    img_diff[mask_inv] = 0
    return img_diff


def approx_contours(contours):
    return [cv2.approxPolyDP(cnt, 3, True) for cnt in contours if cv2.contourArea(cnt) > 30]


def find_diff_cnt(diff_img):
    edges = cv2.Canny(diff_img, 220, 220)
    dilated_img = cv2.dilate(edges, None)
    img, contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CONTOURS_MATCH_I1)
    return contours


def test_diff(image_origin_name):
    img_origin = cv2.imread(os.path.join(images_origin_dataset_path, image_origin_name))
    img_origin = cv2.cvtColor(img_origin, cv2.COLOR_RGB2BGR)
    # distort and change image
    img_changed = cv2.circle(img_origin.copy()[200:, 100:], (1000, 100), 20, (255, 225, 0), thickness=5)
    img_changed = cv2.rotate(img_changed, cv2.ROTATE_180)
    h, w, _ = img_changed.shape
    # img_changed=cv2.resize(img_changed,(w-10,h-10))
    img_changed = attach_image(seal_img, img_changed)
    # resize origin image
    h, w, _ = img_origin.shape
    img_origin = img_origin[200:, 100:]
    img_origin = cv2.resize(img_origin, (int(w // 1.2), int(h // 1.2)))
    plt.figure(figsize=(32, 32))
    plt.subplot(1, 3, 1)
    plt.imshow(img_origin)
    plt.subplot(1, 3, 2)
    plt.imshow(img_changed)
    # find the difference
    image_aligned, dst_polygon = align_image(img_origin, img_changed)
    diff_img = diff_image(img_origin, image_aligned, dst_polygon)
    contours = find_diff_cnt(diff_img)
    contours_img = cv2.drawContours(image_aligned.copy(), approx_contours(contours), -1, (0, 255, 0), thickness=2)
    plt.subplot(1, 3, 3)
    plt.imshow(contours_img)
    plt.show()


if __name__ == '__main__':
    images_origin_dataset_path = './images/images_origin'
    images_changed_dataset_path = './images/images_changed'
    seal_image_path = './images/obj1.png'
    image_origin_names = os.listdir(images_origin_dataset_path)

    # load image
    seal_img = cv2.imread(seal_image_path, -1)
    seal_img = cv2.cvtColor(seal_img, cv2.COLOR_RGB2BGR)
    for image_origin_name in image_origin_names:
        test_diff(image_origin_name)
