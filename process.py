import cv2
import numpy as np

def preprocess_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 42, 89)
    kernel = np.ones((3, 3))
    dial = cv2.dilate(canny, kernel=kernel, iterations=2)

    return dial

def findContours(img, original, draw=False):
    # find the set of contours on the threshed image
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # sort them by highest area
    proper = sorted(contours, key=cv2.contourArea, reverse=True)

    four_corners_set = []

    for cnt in proper:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)

        # only select those contours with a good area
        if area > 10000:
            # find out the number of corners
            approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, closed=True)
            numCorners = len(approx)

            if numCorners == 4:
                # create bounding box around shape
                x, y, w, h = cv2.boundingRect(approx)

                if draw:
                    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # make sure the image is oriented right: top left, bot left, bot right, top right
                l1 = np.array(approx[0]).tolist()
                l2 = np.array(approx[1]).tolist()
                l3 = np.array(approx[2]).tolist()
                l4 = np.array(approx[3]).tolist()

                finalOrder = []

                # sort by X vlaue
                sortedX = sorted([l1, l2, l3, l4], key=lambda x: x[0][0])

                # sortedX[0] and sortedX[1] are the left half
                finalOrder.extend(sorted(sortedX[0:2], key=lambda x: x[0][1]))

                # now sortedX[1] and sortedX[2] are the right half
                # the one with the larger y value goes first
                finalOrder.extend(sorted(sortedX[2:4], key=lambda x: x[0][1], reverse=True))

                four_corners_set.append(finalOrder)

                if draw:
                    for a in approx:
                        cv2.circle(original, (a[0][0], a[0][1]), 10, (255, 0, 0), 3)

    return four_corners_set

def flatten_card(img, set_of_corners):
    width, height = 200, 300
    img_outputs = []

    for i, corner_set in enumerate(set_of_corners):
        # get the 4 corners of the card
        pts1 = np.float32(corner_set)
        # now define which corner we are referring to
        pts2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

        # transformation matrix
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        imgOutput = cv2.warpPerspective(img, matrix, (width, height))

        img_outputs.append(imgOutput)

    return img_outputs
