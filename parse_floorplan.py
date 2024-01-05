import json
import sys
import os
import cv2
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

direction = {"north": "北", 
             "east": "东",
             "south": "南",
             "west": "西",
             "north-west": "西北",
             "south-west": "西南",
             "north-east": "东北",
             "south-east": "东南",
             }


def main():
    infile = sys.argv[1]
    img_file = sys.argv[2]

    img = cv2.imread(img_file)
    img_h, img_w, img_c = img.shape
    print(img.shape)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 对图像进行开运算

    _, thresholded_image = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.imwrite("thresholded_image.jpg", thresholded_image)

    # 定义结构元素
    kernel_size = (3, 3)
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    #eroded_img = cv2.dilate(thresholded_image, structuring_element)
    eroded_img = cv2.dilate(thresholded_image, structuring_element, iterations=1)
    #cv2.imwrite("eroded_img.jpg", eroded_img)

    V = cv2.__version__.split(".")[0]
    if V == "3":
        _, contours, hierarchy = cv2.findContours(eroded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #contours, hierarchy = cv2.findContours(eroded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    elif V == "4": 
        contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_rect = {"area": 0, "rect": None}
    for cont in contours:
        # 外接矩形
        r_x, r_y, r_w, r_h = cv2.boundingRect(cont)
        r_area = r_w * r_h
        if r_area > max_rect["area"] and abs(img_h * img_w - r_area) > 400:
            #print(abs(img_h * img_w - r_area))
            max_rect["area"] = r_area
            max_rect["rect"] = (r_x, r_y, r_w, r_h)
        #cv2.rectangle(img, (r_x, r_y), (r_x + r_w,  r_y + r_h), (255, 0, 255), 10)
        #cv2.rectangle(img, (r_x + 20, r_y + 20), (r_x + r_w - 10,  r_y + r_h - 10), (255, 0, 255), 10)
            
    # 在原图上画出预测的矩形
    print("max_rect={}".format(max_rect["rect"]))
    cv2.rectangle(img, (max_rect["rect"][0], max_rect["rect"][1]), (max_rect["rect"][0] + max_rect["rect"][2], max_rect["rect"][1] + max_rect["rect"][3]), (255, 0, 0), 10)

    with open(infile, "r") as f:
        data = json.load(f)
    points = data["floorplans"][0]["points"]
    
    vector_center_x, vector_center_y = int(img_w / 2), int(img_h / 2)
    #print(vector_center_x, vector_center_y)
    cv2.circle(img, (vector_center_x, vector_center_y), 30, (0, 128, 0), thickness=-1)

    min_p_x = float("inf")
    max_p_x = float("-inf")
    min_p_y = float("inf")
    max_p_y = float("-inf")
    #points = {p["id"] : {"x": p["x"], "y": p["y"]} for p in points}
    new_points = {}
    for p in points:
        #new_points[p["id"]] = {"x": p["x"], "y": p["y"]}
        #x = int(p["x"] / scale)
        #y = int(p["y"] / scale)
        x = p["x"]
        y = p["y"]
        new_points[p["id"]] = {"x": x, "y": y}
        min_p_x = min(x, min_p_x)
        max_p_x = max(x, max_p_x)
        min_p_y = min(y, min_p_y)
        max_p_y = max(y, max_p_y)

    raster_center_x = int((min_p_x + max_p_x) / 2)
    raster_center_y = int((min_p_y + max_p_y) / 2)
    raster_fg_w = max_p_x - min_p_x 
    raster_fg_h = max_p_y - min_p_y 

    points = {}
    idx = 0

    scale = 10 
    scale_x = raster_fg_w / max_rect["rect"][2]
    scale_y = raster_fg_h / max_rect["rect"][3]
    scale = max(scale_x, scale_y)
    #print(raster_fg_w, raster_fg_h, max_rect["rect"])
    #print("scale_x={}, scale_y={}".format(scale_x, scale_y))
    font = ImageFont.truetype("SimSun.ttf", 36)
    for p_id, p in new_points.items():
        x = int(vector_center_x + (p["x"] - raster_center_x) / scale)
        y = int(vector_center_y - (p["y"] - raster_center_y) / scale)
        #y = int(vector_center_y + (p["y"] - center_y) / scale)
        #x = int(vector_center_x - (p["x"] - center_x) / scale)
        points[p_id] = {"x": x, "y": y}
        #print(idx, p["x"], p["y"], x, y)
        # 设置要添加的文本内容、位置和样式

        # 将文本添加到图像上
        cv2.putText(img, str(idx), (x - 20, y - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
        cv2.circle(img, (x, y), 20, (0, 0, 255), thickness=-1)
        idx += 1

    lines = data["floorplans"][0]["lines"]
    lines = {l["id"] : {"points": [points[p] for p in l["points"]]} for l in lines}
    areas = data["floorplans"][0]["areas"]
    k_v_areas = {a["id"] : a["roomName"] for a in data["floorplans"][0]["areas"]}
    out_json = []
    for item in areas:
        room_points = [points[p] for p in item["points"]]
        room_name = item["roomName"] 
        if "sizeWithoutLine" in item:
            room_size = "{:.1f}平方米".format(item["sizeWithoutLine"] / 1000000) 
        else:
            room_size = "{:.1f}平方米".format(item["size"] / 1000000) 
        attachments = item["attachments"] 
        #room_lines = [lines[l["id"]] for l in attachments["lineItems"]]
        room_lines = [lines[l["id"]] for l in attachments["lines"]]
        near_rooms = [{direction[a["direction"]]: k_v_areas[a["id"]]} for a in attachments["areas"]]
        out_json.append({
            "room_name": room_name,
            "room_size": room_size,
            "room_points": room_points,
            "room_lines": room_lines,
            "near_rooms": near_rooms
            })
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        pts = np.array([[p["x"], p["y"]] for p in room_points], np.int32) 
        cv2.polylines(img, [pts], True, color, thickness=10)
        cv2.fillPoly(img, [pts], color)
        for line in room_lines:
            p = line["points"]
            start_point = (p[0]["x"], p[0]["y"])
            end_point = (p[1]["x"], p[1]["y"])
            cv2.line(img, start_point, end_point, color, 10)

    r_w = max_p_x - min_p_x
    r_h = max_p_y - min_p_y

    #print(min_p_x, min_p_y, max_p_x, max_p_y, r_w, r_h, r_w / (2050 - 340), r_h / (1550 - 280))
    with open("std_floorplan_sample.json", "w") as f:
        json.dump(out_json, f, indent=4, ensure_ascii=False)
    cv2.imwrite("floorplan_show.png", img)
    return

if __name__ == "__main__":
    main()
