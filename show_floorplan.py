import cv2
import os
import pandas as pd
import sys
import numpy as np
import json
from tqdm import tqdm



def main():
    csv_file = sys.argv[1]
    save_path = "r2v_out"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df = pd.read_csv(csv_file, sep="\t")

    max_x = 0
    max_y = 0
    for index, row in df.iterrows():

        frame_dict = json.loads(row["frame_dict"])
        width = 224
        height = 224
        #img = np.ones((width, height, 4), dtype=np.uint8)
        img = np.zeros((width, height, 4), dtype=np.uint8)
        #img *= 255 # white background
        area = frame_dict["area"]
        for item in area:
            for p in item["pts"]:
                max_x = max(p[0], max_x)
                max_y = max(p[1], max_y)
            pts = np.array(item["pts"], np.int32)
            color = [int(255 * c) for c in item["color"]]
            color = [color[2], color[1], color[0], color[3]]
            #color = (0, 255, 0)
            #cv2.polylines(img, [pts], True, color, thickness=2)
            cv2.polylines(img, [pts], True, color, thickness=0)
            cv2.fillPoly(img, [pts], color)
        wall = frame_dict["wall"]
        for item in wall:
            pts = item["pts"]
            for p in item["pts"]:
                max_x = max(p[0], max_x)
                max_y = max(p[1], max_y)
            #color = item["color"]
            #color = (255, 0, 0)
            color = [int(255 * c) for c in item["color"]]
            color = [color[2], color[1], color[0], color[3]]
            #color = [255, 255, 255, 255]
            #color = (0, 255, 0)
            cv2.line(img, pts[0], pts[1], color, thickness=2)

        wall_item = frame_dict["wall_item"]
        for item in wall_item:
            pts = item["pts"]
            for p in item["pts"]:
                max_x = max(p[0], max_x)
                max_y = max(p[1], max_y)
            #color = item["color"]
            #color = (0, 0, 255)
            color = [int(255 * c) for c in item["color"]]
            color = [color[2], color[1], color[0], color[3]]
            #if color[0] == 25:
            #    color = [0, 255, 255, 255]
            #    print(color)
            cv2.line(img, pts[0], pts[1], color, thickness=2)
        print(max_x, max_y)

        #print(row)
        cv2.imwrite(os.path.join(save_path, "{}.png".format(row["frame_id"])), img)

    return

if __name__ == "__main__":
    main()
