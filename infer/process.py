#! python3
# -*- encoding: utf-8 -*-

import cv2
import numpy as np


def crop_flow(flow):
    mask = np.all(flow >= -1, axis=-1) & np.all(flow <= 1, axis=-1)
    max_nonzero_ratio = 0.4
    max_crop_ratio = 0.1
    mask_h, mask_w = mask.shape[0], mask.shape[1]
    y0 = int(mask_h * max_crop_ratio)
    for i in range(0, int(mask_h * max_crop_ratio)):
        if np.count_nonzero(mask[i]) / mask_w > max_nonzero_ratio:
            y0 = i
            break

    y1 = mask_h - 1 - int(mask_h * max_crop_ratio)
    for i in range(mask_h - 1, y1, -1):
        if np.count_nonzero(mask[i]) / mask_w > max_nonzero_ratio:
            y1 = i
            break

    crop_mask = mask[y0:y1]
    mask_h, mask_w = crop_mask.shape[0], crop_mask.shape[1]
    x0 = int(mask_w * max_crop_ratio)
    for i in range(0, x0):
        if np.count_nonzero(mask[:, i]) / mask_h > max_nonzero_ratio:
            x0 = i
            break

    x1 = mask_w - 1 - int(mask_w * max_crop_ratio)
    for i in range(mask_w - 1, x1, -1):
        if np.count_nonzero(mask[:, i]) / mask_h > max_nonzero_ratio:
            x1 = i
            break
    flow = flow[y0:y1, x0:x1]
    return flow


def quantization_color(image, mask, edge):
    color = np.asarray([(0, 255, 0), (0, 0, 255)])
    mask_img = np.zeros_like(image)
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    ret_img = lab_img.copy()
    _, label, stats, _ = cv2.connectedComponentsWithStats(edge, connectivity=8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    kmeans_k = 2
    color_dist_th = 25
    cc_cnt_th = 0.06

    for i, cc in enumerate(stats):
        if i == 0:
            continue
        x, y, w, h, area = cc[0], cc[1], cc[2], cc[3], cc[4]
        crop_label = label[y : y + h, x : x + w]
        crop_lab_img = lab_img[y : y + h, x : x + w]
        crop_ret_img = ret_img[y : y + h, x : x + w]
        crop_mask_img = mask_img[y : y + h, x : x + w]
        label_i = np.argwhere(crop_label == i)
        pixels = crop_lab_img[label_i[:, 0], label_i[:, 1]]
        pixels = np.float32(pixels)
        _, k_label, center = cv2.kmeans(
            pixels, kmeans_k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        dist = np.linalg.norm(center[0] - center[1])
        cnt1, cnt2 = np.count_nonzero(k_label == 0), np.count_nonzero(k_label == 1)
        if dist < color_dist_th:
            continue

        min_idx = 1 if max(cnt1, cnt2) == cnt1 else 0
        max_idx = 0 if max(cnt1, cnt2) == cnt1 else 1
        k_mask = np.zeros((h, w), np.uint8)
        low_idx = (k_label == min_idx)[:, 0]
        k_mask[label_i[low_idx, 0], label_i[low_idx, 1]] = 255
        crop_mask_img[label_i[low_idx, 0], label_i[low_idx, 1]] = color[1]
        high_idx = (k_label == max_idx)[:, 0]
        crop_mask_img[label_i[high_idx, 0], label_i[high_idx, 1]] = color[0]

        _, k_label, k_stats, _ = cv2.connectedComponentsWithStats(
            k_mask, connectivity=8
        )
        replace_mask = np.zeros((h, w))
        for k, k_cc in enumerate(k_stats):
            if k == 0:
                continue
            k_area = k_cc[4]
            if k_area / area < cc_cnt_th:
                k_label_i = np.argwhere(k_label == k)
                avg_lab_color = np.mean(
                    crop_lab_img[k_label_i[:, 0], k_label_i[:, 1]], axis=(0, 1)
                )
                color_df = np.linalg.norm(avg_lab_color - center[max_idx])
                if color_df > color_dist_th:
                    replace_mask[k_label_i[:, 0], k_label_i[:, 1]] = 255
                    crop_mask_img[k_label_i[:, 0], k_label_i[:, 1]] = color[0]

        kernel = np.ones((3, 3), np.uint8)
        replace_mask = cv2.morphologyEx(replace_mask, cv2.MORPH_DILATE, kernel)
        crop_ret_img[replace_mask != 0] = center[max_idx]

    return cv2.cvtColor(ret_img, cv2.COLOR_LAB2BGR)


def inpaint_img(image, flow, flags=cv2.INPAINT_TELEA):
    max_work_len = 640
    img_w, img_h = image.shape[1], image.shape[0]
    ratio = max_work_len / max(img_w, img_h)

    work_img = cv2.resize(
        image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA
    )
    work_flow, work_mask = align_flow(work_img, flow)
    if np.count_nonzero(work_mask) == 0:
        return image

    work_img = cv2.remap(work_img, work_flow.astype(np.float32), None, cv2.INTER_LINEAR)

    kernel = np.ones((5, 5), np.uint8)
    gradient = cv2.morphologyEx(work_mask, cv2.MORPH_GRADIENT, kernel)
    edge = ~work_mask & gradient
    quant_img = quantization_color(work_img, work_mask, edge)

    bm_inpaint = cv2.inpaint(quant_img, work_mask, 3, flags)
    return cv2.resize(bm_inpaint, (img_w, img_h))


def align_flow(image, flow):
    img_w, img_h = image.shape[1], image.shape[0]
    bm_flow = flow / 2 + 0.5
    bm_flow[..., 0] = bm_flow[..., 0] * img_w
    bm_flow[..., 1] = bm_flow[..., 1] * img_h
    if bm_flow.shape[0] != img_h or bm_flow.shape[1] != img_w:
        bm_flow = cv2.resize(bm_flow, (img_w, img_h))

    flow_mask = ~cv2.inRange(bm_flow, (0, 0), (img_w - 1, img_h - 1))
    return bm_flow, flow_mask


def pre_process(image, target_size):
    """
    压缩图像尺寸
    """
    resize = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    resize = resize.astype(np.float32) / 255.0 - 0.5
    resize = np.transpose(resize, [2, 0, 1])
    return resize, image


def post_process(in_image, bm_flow, with_crop=True):
    """
    bm flow is -1 ~ 1 3dim numpy array
    in_image is 0 - 255
    """
    print(f"shape of bm: {bm_flow.shape}")
    in_image = (in_image * 255).astype(np.uint8)
    # print(in_image)
    if bm_flow.shape[0] == 2:
        bm_flow = np.transpose(bm_flow, [1, 2, 0])

    min_val = np.min(bm_flow)
    max_val = np.max(bm_flow)
    # bm_flow = -1 + 2 * (bm_flow - min_val) / (max_val - min_val)
    # print(bm_flow)
    bm_flow = ((bm_flow / 448) - 0.5) * 2.0
    print(min_val, max_val)
    if with_crop:
        bm_flow = crop_flow(bm_flow)
    print(f"shape of bm: {bm_flow.shape}, img: {in_image.shape}, {in_image.dtype}")

    dewarp_inpaint = inpaint_img(in_image, bm_flow)

    bm_flow, flow_mask = align_flow(in_image, bm_flow)

    dewarp_raw = cv2.remap(
        in_image,
        bm_flow.astype(np.float32),
        None,
        cv2.INTER_LINEAR,
        borderValue=(255, 0, 255),
    )

    dewarp_tmp = dewarp_raw.copy()
    dewarp_tmp[flow_mask > 0] = dewarp_inpaint[flow_mask > 0]

    return dewarp_tmp, dewarp_raw, bm_flow
