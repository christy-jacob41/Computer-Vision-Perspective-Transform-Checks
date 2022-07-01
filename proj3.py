import argparse
import os
import cv2
import numpy as np


def rectify(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

def process_img(img_path):
    frame_orig = cv2.imread(img_path)
    height = frame_orig.shape[0]
    width = frame_orig.shape[1] 
    if height > 1200 or width > 1200:
        frame_orig = cv2.resize(frame_orig, (int(width/5), int(height/5)))
    height = frame_orig.shape[0]
    width = frame_orig.shape[1] 
    if height < 200 or width < 200:
        frame_orig = cv2.resize(frame_orig, (int(width*4), int(height*4)))
    ### Replace the code below to show only the check and apply transform.
    
    frame_result = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)


    gray = cv2.cvtColor(frame_result, cv2.COLOR_BGR2GRAY)
    
    
    # kernel = np.ones((5, 5), np.uint8) # Reduce Noise Of Image
    # erosion = cv2.erode(gray, kernel, iterations=1)
    # opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # edged = cv2.Canny(closing, 10, 100)


    # cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    # screenCnt = cv2.approxPolyDP(cnts[0], 0.01 * cv2.arcLength(cnts[0], True), True)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 10, 100)
    kernel = np.ones((3,3))
    imgDil = cv2.dilate(edges, kernel, iterations =2)
    imgErode = cv2.erode(imgDil, kernel, iterations =1)
    
    imgCnts = frame_orig.copy()
    cnts, h = cv2.findContours(imgErode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cntList = []
    for i in range(len(cnts)):
        hull = cv2.convexHull(cnts[i])
        cntList.append(hull)
    cv2.drawContours(imgCnts, cntList, -1, (0, 255, 0), 5)

    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

    screenCnt = cv2.approxPolyDP(cnts[0], 0.05 * cv2.arcLength(cnts[0], True), True)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    rotateCheck = True
    if cv2.contourArea(screenCnt) < 15000:
        rotateCheck = False
        screenCnt = np.int32([[frame_orig.shape[1]*0.2,frame_orig.shape[0]*0.3],[frame_orig.shape[1]*0.8,frame_orig.shape[0]*0.3],[frame_orig.shape[1]*0.8,frame_orig.shape[0]*0.7],[frame_orig.shape[1]*0.2,frame_orig.shape[0]*0.7]])

    minAreaRect = cv2.minAreaRect(screenCnt)
    cv2.drawContours(frame_result, [screenCnt], -1, (0, 255, 0), 5)

    approx = rectify(screenCnt)
    pts2 = np.float32([[0,0],[frame_orig.shape[1],0],[frame_orig.shape[1],frame_orig.shape[0]],[0,frame_orig.shape[0]]])
    M = cv2.getPerspectiveTransform(approx,pts2)
    result = cv2.warpPerspective(frame_orig,M,(frame_orig.shape[1],frame_orig.shape[0]))
    if minAreaRect[-1] > 81 and rotateCheck:
        result = cv2.rotate(result, cv2.cv2.ROTATE_90_CLOCKWISE)


    ### Replace the code above.
    cv2.imshow("Original", frame_orig)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Check prepartion project")
    parser.add_argument('--input_folder', type=str, default='samples', help='check images folder')
    
    args = parser.parse_args()
    input_folder = args.input_folder
   
    for check_img in os.listdir(input_folder):
        img_path = os.path.join(input_folder, check_img)
        if img_path.lower().endswith(('.png','.jpg','.jpeg', '.bmp', '.gif', '.tiff')):
            process_img(img_path)
            