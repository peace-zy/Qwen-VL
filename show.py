import cv2
import sys

def main():
    infile = sys.argv[1]
    img = cv2.imread(infile)
    """
    re_img = cv2.resize(img, (256, 256))
    h, w, c = img.shape
    h, w, c = re_img.shape
    print(re_img.shape)
    pos = [0.44, 0.37, 0.56, 0.51]
    pos = [0.42, 0.28, 0.64, 0.65]
    ratio = h / w
    #cv2.rectangle(img, (int(w * 0.44), int(h * 0.37)), (int(w * 0.56), int(h * 0.51)), (0, 255, 0), 2)
    center_x = pos[0] * w
    center_y = pos[1] * h
    b_w = pos[2] * w
    b_h = pos[3] * h
    #cv2.rectangle(img, (int(w * pos[0]), int(h * pos[1])), (int(w * pos[2]), int(h * pos[3])), (0, 0, 255), 2)
    cv2.rectangle(img, (int(w * pos[0]), int(h * pos[1])), (int(w * pos[2]), int(h * pos[3])), (0, 0, 255), 2)
    #cv2.rectangle(img, (int(w * ratio * pos[0]), int(h * pos[1])), (int(w * ratio * pos[2]), int(h * pos[3])), (255, 0, 0), 2)
    #cv2.rectangle(img, (int(center_x - b_w / 2), int(center_y - b_h / 2)), (int(center_x + b_h / 2), int(center_y + b_h / 2)), (0, 255, 0), 2)
    cv2.imwrite("show.jpg", img)
    """
    #客厅
    cv2.rectangle(img, (325, 264), (1094, 1534), (0, 0, 255), 2)
    #阳台
    cv2.rectangle(img, (1076, 264), (1406, 444), (0, 0, 255), 2)
    #卧室
    cv2.rectangle(img, (1406, 264), (2046, 912), (0, 0, 255), 2)
    #厨房
    cv2.rectangle(img, (1079, 450), (1397, 909), (0, 0, 255), 2)
    #洗手间
    cv2.rectangle(img, (1289, 1084), (1596, 1273), (0, 0, 255), 2)
    #卧室
    cv2.rectangle(img, (1599, 909), (1911, 1282), (0, 0, 255), 2)
    cv2.imwrite("show.jpg", img)



    return

if __name__ == "__main__":
    main()
