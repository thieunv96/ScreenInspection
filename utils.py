import numpy as np
import cv2


def split_image(img, target_size=(224,224)):
    org_h, org_w, _ = img.shape
    w, h = target_size
    step_w = org_w // w if org_w % w == 0 else org_w // w + 1
    step_h = org_h // h if org_h % h == 0 else org_h // h + 1
    print(step_w, step_h)
    boxs = []
    sub_images = []
    for s_y in range(step_h):
        for s_x in range(step_w):
            x = s_x * w if s_x < step_w - 1 else org_w - w
            y = s_y * h if s_y < step_h - 1 else org_h - h
            box = [x, y, w, h]
            sub_img = img[y:y+h, x:x+w]
            boxs.append(box)
            sub_images.append(sub_img)

    return np.array(sub_images), np.array(boxs)