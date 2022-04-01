# TODO: programatic myData parsing
# TODO: send roi data from regionselector script to txt file that this script reads
# TODO: pattern matching to recognize flipped, rotated forms etc
# TODO: PRE PROCESSING ALGORITH!!!!

import cv2
import numpy
from pytesseract import pytesseract
import re
import ast
import os

pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
name_regex = '[A-Z]+[,\.]+\s*[A-Z]+'
npi_regex = '[0-9]{10}'

# Takes a cv2 image and pre-processes it in preparation for Tesseract OCR.
def pre_process_form(form):

    form_width = form.shape[1]
    form_height = form.shape[0]
    # TODO: Breakpoint for if the image is already big enough, and computation to find scale value
    # to upscale the image to some consistent size.
    dim = (form_width * 6, form_height * 6)

    # Upscale the image to 2x its original size
    form = cv2.resize(form, dim, interpolation = cv2.INTER_CUBIC)
    
    form = cv2.cvtColor(form, cv2.COLOR_BGR2GRAY)
    # dilation and erosion to reduce noise
    kernel = numpy.ones((1, 1), numpy.uint8)
    form = cv2.dilate(form, kernel, iterations=1)
    form = cv2.erode(form, kernel, iterations=1)
    
    # blur
    # form = cv2.threshold(cv2.GaussianBlur(form, (5, 5), 0), 0, 255, cv2.THRESH_OTSU)[1]
    # form = cv2.threshold(cv2.bilateralFilter(form, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.adaptiveThreshold(cv2.bilateralFilter(form, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    # cv2.adaptiveThreshold(cv2.medianBlur(form, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    
    # 4-neighbors laplacian filter
    kernel = numpy.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    form = cv2.filter2D(form, -1, kernel)

    return form

def process_data(data):
    types = [element[0] for element in data]
    unparsed_data = [element[1] for element in data]
    for i in range(len(types)):
        if types[i] == "Name":
            match = re.search(name_regex, unparsed_data[i])
            if match != None:
                print("Name: " + match.group(0))
            else:
                print("Regex parse failed: " + unparsed_data[i])
    

def ocr_form(template, form):
    h, w, c = template.shape
    orb = cv2.ORB_create(1000000)
    key_point1, descriptor1 = orb.detectAndCompute(template, None)
    # Read roi from roi file
    roi_file = open("roi.txt", "r")
    roi_file_text = roi_file.read()
    roi = ast.literal_eval(roi_file_text)
    roi_file.close()

    myData = []

    key_point2, descriptor2 = orb.detectAndCompute(form, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(descriptor2, descriptor1)
    matches = sorted(matches, key = lambda x: x.distance)
    # Take the top 25% keypoint matches.
    percentage = 25
    good = matches[:int(len(matches) * (percentage / 100))]
    # imgMatch = cv2.drawMatches(form, key_point2, template, key_point1, good, None, flags = 2)
    src_points = numpy.float32([key_point2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = numpy.float32([key_point1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    imgScan = cv2.warpPerspective(form, M, (w, h))
    imgShow = imgScan.copy()
    
    imgMask = numpy.zeros_like(imgShow)
    
    for x,r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0],r[1][1]), (0, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
        
        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        
        if r[2] == 'text' or r[2] == 'number':
            tesseractImage = pre_process_form(imgCrop)
            cv2.imshow(str(r[3]), tesseractImage)
            myData.append((r[3], pytesseract.image_to_string(tesseractImage)))
            cv2.waitKey(0)

    process_data(myData)

    
    
    
