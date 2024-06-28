# This code is implementation of merging foreground and background images
# to create a new image with foreground object on background image.

import os
import cv2
import numpy as np

def merge_fg_bg(fg_img, bg_img, alpha=1):
    fg_img = cv2.resize(fg_img, (bg_img.shape[1], bg_img.shape[0]))
    fg_img = fg_img.astype(np.float32)
    bg_img = bg_img.astype(np.float32)
    fg_img = alpha * fg_img
    bg_img = (1 - alpha) * bg_img
    merged_img = fg_img + bg_img
    merged_img = np.clip(merged_img, 0, 255)
    merged_img = merged_img.astype(np.uint8)
    return merged_img

def arg_parse():
    import argparse
    parser = argparse.ArgumentParser(description='Merge foreground and background images')
    parser.add_argument('--fg_dir', required=True, help='Path to the foreground image')
    parser.add_argument('--bg_dir', required=True, help='Path to the background image')
    parser.add_argument('--output_dir', required=True, help='Path to the output directory')
    parser.add_argument('--alpha', type=float, default=1, help='Alpha value for merging')
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    list_fg = os.listdir(args.fg_dir)
    list_bg = os.listdir(args.bg_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for fg, bg in zip(list_fg, list_bg):
        fg_img = cv2.imread(os.path.join(args.fg_dir, fg))
        bg_img = cv2.imread(os.path.join(args.bg_dir, bg))
        merged_img = merge_fg_bg(fg_img, bg_img, args.alpha)
        cv2.imwrite(os.path.join(args.output_dir, fg), merged_img)