import cv2
import numpy as np

def splitOverlap(image, overlap_percentage):
    h, w, _ = image.shape
    # print(h,w)

    h_sp = h / 2
    w_sp = w / 2
    h_sp = int(h_sp + h_sp * overlap_percentage)
    w_sp = int(w_sp + w_sp * overlap_percentage)


    split_cord = [[0, 0, h_sp, w_sp],
                  [0, w - w_sp, h_sp, w],
                  [h - h_sp, 0, h, w_sp],
                  [h - h_sp, w - w_sp, h, w]]

    res = np.array([
                        image[split_cord[i][0]:split_cord[i][2], split_cord[i][1]:split_cord[i][3], :]
                        for i in range(4)
                   ])

    return res, split_cord

def joinOverlap(images, split_cord):
    if not split_cord: return images

    w = split_cord[-1][-1]
    h = split_cord[-1][-2]

    w_shape = split_cord[0][-1]
    h_shape = split_cord[0][-2]

    image_construct = np.zeros((1, h, w))
    for i, part in enumerate(split_cord):
        image_construct[0, part[0]:part[2], part[1]:part[3]] = cv2.resize(images[i, :, :], (w_shape, h_shape))

    return image_construct

img = cv2.imread("input/sample_1.png")
imgs, sc = splitOverlap(img, 0.1)
# for image in imgs:
#     cv2.imshow("Image", image)
#     cv2.waitKey(0)

join = joinOverlap(imgs, sc)
cv2.imshow("Image", join[0])
cv2.waitKey(0)
