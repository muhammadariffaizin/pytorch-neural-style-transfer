"""
Copyright (C) 2022 HKUST VGD Group
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# Usage:
# cd day2golden_results
# python gen_html.py -i ./results
"""

import argparse
import os
import glob

def append_index(content_dir, style_dir, output_dir, iterations):
    # get current dir
    input_dir = os.getcwd()
    index_path = os.path.join(input_dir, "index.html")

    index = open(index_path, "w")
    index.write("<html><body>")
    index.write("<h1></h1>")
    index.write("<table><tr>")
    index.write("<th>content / style</th>")

    content_files = sorted(glob.glob(os.path.join(content_dir, "*")))
    for i in range(len(content_files)):
        content_files[i] = content_files[i].split('.')[0]
    print("%d content images"%(len(content_files)))

    style_files = sorted(glob.glob(os.path.join(style_dir, "*")))
    for i in range(len(style_files)):
        style_files[i] = style_files[i].split('.')[0]
    print("%d style images"%(len(style_files)))

    for path in style_files:
        index.write("<td><img src='%s/%s.png' width='256'></td>" % (style_dir, os.path.basename(path)))
    index.write("</tr>")

    for path in content_files:
        index.write("<tr>")
        index.write("<td><img src='%s/%s.png' width='256'></td>" % (content_dir, os.path.basename(path)))
        for style_path in style_files:
            index.write("<td><img src='%s/combined_%s_%s/%s.jpg' width='256'></td>" % (output_dir, os.path.basename(path), os.path.basename(style_path), str(iterations).zfill(4)))
        index.write("</tr>")

    return index_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", "-c", default="./", help="path to folder containing content images")
    parser.add_argument("--style_dir", "-s", default="./", help="path to folder containing style images")
    parser.add_argument("--output_dir", "-o", default="./", help="path to folder containing output images")
    parser.add_argument("--iterations", "-i", default=1000, help="number of iterations")
    args = parser.parse_args()

    append_index(args.content_dir, args.style_dir, args.output_dir, args.iterations)
