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
import pandas as pd

def append_index(content_dir, style_dir, output_dir, content_list=None, style_list=None, mode="max", iterations=1000):
    # get current dir
    input_dir = os.getcwd()
    index_path = os.path.join(input_dir, "index.html")

    if content_list is None:
        content_files = sorted(glob.glob(os.path.join(content_dir, "*")))
    else:
        content_files = content_list
    print("%d content images"%(len(content_files)))
    print(content_files)

    if style_list is None:
        style_files = sorted(glob.glob(os.path.join(style_dir, "*")))
    else:
        style_files = style_list
    print("%d style images" % (len(style_files)))

    index = open(index_path, "w")
    index.write("<html>\n")
    index.write("<head><style>table, th, td {border: 1px solid black; border-collapse: collapse;}</style></head>\n")
    index.write("<body>\n")
    index.write("<h1>Overall Results</h1>\n")
    index.write("<table style='border: none;'><tr>\n")

    index.write("<td></td>\n")
    index.write("<td><p>style_file</p></td>\n")
    for path in style_files:
        index.write("<td><p>%s</p></td>\n" % (os.path.basename(path)))
    index.write("</tr>\n")

    index.write("<tr>\n")
    index.write("<td>content_file</td>\n")
    index.write("<th>content / style</th>\n")
    for path in style_files:
        index.write("<td><img src='%s/%s' width='256'></td>\n" % (style_dir, os.path.basename(path)))
    index.write("</tr>\n")

    for path in content_files:
        index.write("<tr>")
        index.write("<td><p>%s<p></td>\n" % (os.path.basename(path)))
        index.write("<td><img src='%s/%s' width='256'></td>\n" % (content_dir, os.path.basename(path)))
        for style_path in style_files:
            if mode == "max":
                output_file_list = glob.glob(os.path.join(output_dir, "combined_%s_%s" % (os.path.basename(path).split('.')[0], os.path.basename(style_path).split('.')[0]), "*.jpg"))
                output_file_list.extend(glob.glob(os.path.join(output_dir, "combined_%s_%s" % (os.path.basename(path).split('.')[0], os.path.basename(style_path).split('.')[0]), "*.png")))
                output_files = sorted(output_file_list)
                if len(output_files) == 0:
                    print("No output files found for %s and %s" % (os.path.basename(path), os.path.basename(style_path)))
                    index.write("<td></td>\n")
                    continue
                output_file = os.path.basename(output_files[-1])
            elif mode == "specific":
                output_file = str(iterations).zfill(4) + ".jpg"
            index.write("<td><img src='%s/combined_%s_%s/%s' width='256'></td>\n" % (output_dir, os.path.basename(path).split('.')[0], os.path.basename(style_path).split('.')[0], output_file))
        index.write("</tr>\n")
    index.write("</table>\n")

    if args.reports:
        index.write("<h1>Loss Reports</h1>\n")
        index.write("<h2>Total Loss</h2>\n")
        index.write("<table>\n<tr>\n")
        index.write("<td></td>\n")
        for path in style_files:
            index.write("<td><p>%s</p></td>\n" % (os.path.basename(path)))
        index.write("</tr>\n")

        index.write("<tr>\n")
        for path in content_files:
            index.write("<td><p>%s</p></td>\n" % (os.path.basename(path)))

            for style_path in style_files:
                loss_file = os.path.join(output_dir, "combined_%s_%s" % (os.path.basename(path).split('.')[0], os.path.basename(style_path).split('.')[0]), "loss.csv")
                if not os.path.exists(loss_file):
                    print("No loss file found for %s and %s" % (os.path.basename(path), os.path.basename(style_path)))
                    index.write("<td></td>\n")
                    continue
                loss_df = pd.read_csv(loss_file)
                index.write("<td><p>%s</p></td>\n" % (loss_df["total_loss"].values[-1]))
            index.write("</tr>\n")
        index.write("</table>\n")

        index.write("<h2>Content Loss</h2>\n")
        index.write("<table>\n<tr>\n")
        index.write("<td></td>\n")
        for path in style_files:
            index.write("<td><p>%s</p></td>\n" % (os.path.basename(path)))
        index.write("</tr>\n")

        index.write("<tr>\n")
        for path in content_files:
            index.write("<td><p>%s</p></td>\n" % (os.path.basename(path)))

            for style_path in style_files:
                loss_file = os.path.join(output_dir, "combined_%s_%s" % (os.path.basename(path).split('.')[0], os.path.basename(style_path).split('.')[0]), "loss.csv")
                if not os.path.exists(loss_file):
                    print("No loss file found for %s and %s" % (os.path.basename(path), os.path.basename(style_path)))
                    index.write("<td></td>\n")
                    continue
                loss_df = pd.read_csv(loss_file)
                index.write("<td><p>%s</p></td>\n" % (loss_df["content_loss"].values[-1]))
            index.write("</tr>\n")
        index.write("</table>\n")

        index.write("<h2>Style Loss</h2>\n")
        index.write("<table>\n<tr>\n")
        index.write("<td></td>\n")
        for path in style_files:
            index.write("<td><p>%s</p></td>\n" % (os.path.basename(path)))
        index.write("</tr>\n")

        index.write("<tr>\n")
        for path in content_files:
            index.write("<td><p>%s</p></td>\n" % (os.path.basename(path)))

            for style_path in style_files:
                loss_file = os.path.join(output_dir, "combined_%s_%s" % (os.path.basename(path).split('.')[0], os.path.basename(style_path).split('.')[0]), "loss.csv")
                if not os.path.exists(loss_file):
                    print("No loss file found for %s and %s" % (os.path.basename(path), os.path.basename(style_path)))
                    index.write("<td></td>\n")
                    continue
                loss_df = pd.read_csv(loss_file)
                index.write("<td><p>%s</p></td>\n" % (loss_df["style_loss"].values[-1]))
            index.write("</tr>\n")
        index.write("</table>\n")

        index.write("<h2>Total Variation Loss</h2>\n")
        index.write("<table>\n<tr>\n")
        index.write("<td></td>\n")
        for path in style_files:
            index.write("<td><p>%s</p></td>\n" % (os.path.basename(path)))
        index.write("</tr>\n")

        index.write("<tr>\n")
        for path in content_files:
            index.write("<td><p>%s</p></td>\n" % (os.path.basename(path)))

            for style_path in style_files:
                loss_file = os.path.join(output_dir, "combined_%s_%s" % (os.path.basename(path).split('.')[0], os.path.basename(style_path).split('.')[0]), "loss.csv")
                if not os.path.exists(loss_file):
                    print("No loss file found for %s and %s" % (os.path.basename(path), os.path.basename(style_path)))
                    index.write("<td></td>\n")
                    continue
                loss_df = pd.read_csv(loss_file)
                index.write("<td><p>%s</p></td>\n" % (loss_df["tv_loss"].values[-1]))
            index.write("</tr>\n")
        index.write("</table>\n")

    index.write("</body></html>\n")

    return index_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", "-c", default="./", required=True, help="path to folder containing content images")
    parser.add_argument("--style_dir", "-s", default="./", required=True, help="path to folder containing style images")
    parser.add_argument("--output_dir", "-o", default="./", required=True, help="path to folder containing output images")
    parser.add_argument("--content_list", "-cl", metavar='N', type=str, nargs='+', help="list of content images")
    parser.add_argument("--style_list", "-sl", metavar='N', type=str, nargs='+', help="list of style images")
    parser.add_argument("--reports", "-r", action='store_true', help="generate reports")

    parser.add_argument("--mode", "-m", default="max", required=True, help="max or specific", choices=["max", "specific"])
    parser.add_argument("--iterations", "-i", default=1000, help="number of iterations")
    args = parser.parse_args()

    if args.mode == "max":
        append_index(args.content_dir, args.style_dir, args.output_dir, args.content_list, args.style_list, args.mode)
    elif args.mode == "specific" and args.iterations is None:
        print("Please specify number of iterations")
    elif args.mode == "specific" and args.iterations is not None:
        append_index(args.content_dir, args.style_dir, args.output_dir, args.content_list, args.style_list, args.mode, args.iterations)
