import cv2
import os
import argparse
import ocr

def avg(list):
    return sum(list)/len(list)

def find_relevant_document(template_path, folder_path, ocr_flag):
    matches_list = []
    # Template image
    template = cv2.imread(template_path)
    h, w, c = template.shape
    orb = cv2.ORB_create(1000)
    key_point1, descriptor1 = orb.detectAndCompute(template, None)

    #Images to compare
    pictures = os.listdir(folder_path)
    print(pictures)
    
    # How do I find which image had the BEST keypoint matches
    for index, picture in enumerate(pictures):
        img = cv2.imread(folder_path + "/" + picture)
        key_point2, descriptor2 = orb.detectAndCompute(img, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(descriptor2, descriptor1)
        matches = sorted(matches, key = lambda x: x.distance)
        good = matches[:int(len(matches) * (10 / 100))]
        average = avg([match.distance for match in good])
        matches_list.append((picture, average))

    matches_list = sorted(matches_list, key = lambda x: x[1])
    print(matches_list)
    original_file_name = matches_list[0][0]
    original_file_ext = os.path.splitext(original_file_name)[1]
    old_full_path = folder_path + "/" + original_file_name
    new_full_path = folder_path + "/" + '_relevant_doc' + original_file_ext
    os.rename(old_full_path, new_full_path)

    if ocr_flag == 'y':
        form = cv2.imread(new_full_path)
        ocr.ocr_form(template, form)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finds the closest matching document to the template.")
    parser.add_argument('template', type=str, help="Path to the template image to compare against (required)")
    parser.add_argument('documents', type=str, help="Path to a folder of documents to run the script on (required)")
    parser.add_argument('ocr', type=str, nargs='?', default="n", help="Do you want the form to be ocr'd? (y/n) (optional)")
    args = parser.parse_args()
    find_relevant_document(args.template, args.documents, args.ocr)

        
    

    
    

    
    
