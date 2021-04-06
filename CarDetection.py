import cv2
import numpy
import matplotlib
from ObjectTracker import CentroidTracker


def rescale_frame(frame, percent=.75):
    width = int(frame.shape[1] * percent)
    height = int(frame.shape[0] * percent)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def filter_mask(img, a=None):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(opening, kernel, iterations=1)
    retval, threshold = cv2.threshold(opening, 155, 255, cv2.THRESH_BINARY)
    #threshold = cv2.GaussianBlur(threshold, (5, 5), 1)

    return threshold


def detect_vehicles(fg_mask, min_contour_width=20, min_contour_height=30):

    matches = []

    # finding external contours
    contours, hierarchy = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filtering by with, height
    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height)
        if not contour_valid:
            continue

        # getting center of the bounding box
        centroid = get_centroid(x, y, w, h)

        matches.append(((x, y, w, h), centroid))

    return matches


def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return (cx, cy)


try:
    vidFile = cv2.VideoCapture('AlibiShort.mp4')
except:
    print("problem opening input stream")

if not vidFile.isOpened():
    print("capture stream not open")

size = 0.6
x1 = int(612*size)
x2 = int(1510*size)
y1 = int(210*size)
y2 = int(938*size)


fgbg = cv2.createBackgroundSubtractorKNN(dist2Threshold=1000.0)

nFrames = int(vidFile.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vidFile.get(cv2.CAP_PROP_FPS)


ret, frame = vidFile.read()
h, w = frame.shape[:2]
det_zone = (100, 330, 600, 430)
print(det_zone)

ct = CentroidTracker(detection_zone=det_zone)
while ret:
    patch = rescale_frame(frame, percent=size)
    fgmask = fgbg.apply(patch)
    filtered_img = filter_mask(fgmask)

    squares = detect_vehicles(filtered_img)

    cars = ct.update(squares)

    for key, car in cars.items():
        start_point = (car[0][0], car[0][1])
        end_point = (car[0][0]+car[0][2], car[0][1]+car[0][3])
        patch = cv2.rectangle(patch, start_point, end_point, color=(100, 45, 255), thickness=2)
        patch = cv2.putText(patch, f'carId:{key}', start_point, fontFace=4, fontScale=0.6, color=(0, 0, 0), thickness=1)

    patch = cv2.putText(patch, f'Car passed:{ct.count}', (20, 20), fontFace=4, fontScale=0.6, color=(0, 0, 0), thickness=1)
    patch = cv2.rectangle(patch, (det_zone[0], det_zone[1]), (det_zone[2], det_zone[3]), color=(78, 252, 3), thickness=1)
    cv2.imshow("frameWindow", patch)
    #cv2.imshow("fgmask", filtered_img)

    cv2.waitKey(int(1000/fps))
    ret, frame = vidFile.read()
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
