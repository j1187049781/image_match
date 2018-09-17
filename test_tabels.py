# coding: utf-8


from __future__ import absolute_import, print_function, division
from diff import *

def test_diff(image_origin,img_changed):
    # find the difference
    image_aligned, dst_polygon = align_image(img_origin, img_changed)
    diff_img = diff_image(img_origin, image_aligned, dst_polygon)
    contours = find_diff_cnt(diff_img)
    contours_img = cv2.drawContours(image_aligned.copy(), approx_contours(contours), -1, (0, 255, 0), thickness=2)
    return contours_img

if __name__ == '__main__':
    images_dataset_path = './images/tables'
    seal_image_path = './images/obj1.png'
    image_origin_names = os.listdir(images_dataset_path)

    # load image
    seal_img = cv2.imread(seal_image_path, cv2.IMREAD_COLOR)
    seal_img = cv2.cvtColor(seal_img, cv2.COLOR_BGR2RGB)

    img_origin = cv2.imread(os.path.join(images_dataset_path, '工作周报1.jpg'),cv2.IMREAD_COLOR)
    img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)

    img_changed = cv2.imread(os.path.join(images_dataset_path, '工作周报2.jpg'),cv2.IMREAD_COLOR)
    img_changed = cv2.cvtColor(img_changed, cv2.COLOR_BGR2RGB)
    img_changed = attach_image(seal_img, img_changed)
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(img_origin)
    plt.subplot(1, 3, 2)
    plt.imshow(img_changed)

    contours_img=test_diff(img_origin,img_changed)
    plt.subplot(1, 3, 3)
    plt.imshow(contours_img)
    plt.show()

    img_changed = cv2.cvtColor(img_changed, cv2.COLOR_RGB2BGR)
    cv2.imwrite('img_changed.jpg',img_changed)

    contours_img = cv2.cvtColor(contours_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('diff.jpg',contours_img)