"""sanitize.py 
usage: sanity check if all the json files created from LabelMe software are error-free before converting to any label format
that your object detector requires as input
developer: Seungwon Oh
e-mail: aspiringtechsavvy@gmail.com
"""
import argparse
import glob, json, os
import numpy as np
from labelconverter import labelme_to_coco, labelme_to_yolo # need to have labelconverter.py in the same path


def get_center(obj_points):
    """calculates the center point of the polygon by using the labeled coordinates of a chicken

    :params 
        - obj_points: np.array(8,2), labeled coordinates of the polygon of a chicken
    """  
    x = (np.min(obj_points[:,0]) + np.max(obj_points[:,0])) / 2.0
    y = (np.min(obj_points[:,1]) + np.max(obj_points[:,1])) / 2.0

    return {'x':x,'y':y}

def calc_angle(obj_center, obj_points):
    """find the angle between the line connecting center point and one point and the x-axis in degrees
    
    :param 
        - obj_center: dict, the center coordinate of the polygon calculated by using the labeled coordinates of a chicken
        - obj_points: list, the 1st coordinate of the polygon which indicates the head of a chicken
    """

    # the tail is in the 4th point when partially visible (6 points)
    if len(obj_points) == 6:
        tail_point = (obj_points[3][0], obj_points[3][1])

    # the tail is in the 5th point when fully visible (8 points)
    if len(obj_points) == 8:
        tail_point = (obj_points[4][0], obj_points[4][1])

    # the tail point is in 1th-quadrant
    if obj_center['x'] < tail_point[0] and obj_center['y'] >= tail_point[1]:
        tanx = (obj_center['y'] - tail_point[1]) / (tail_point[0] - obj_center['x'])
        deg = np.arctan(tanx)*57.2958

    # the tail point is in 2nd-quadrant
    if obj_center['x'] >= tail_point[0] and obj_center['y'] > tail_point[1]:
        if obj_center['x'] > tail_point[0]:
            tanx = abs(obj_center['y'] - tail_point[1]) / abs(obj_center['x'] - tail_point[0])
            deg = 180 - np.arctan(tanx)*57.2958

        # when x coordinate of the obj_center and the tail point are the same
        else:
            deg = 90

    # the tail point is in 3rd-quadrant
    if obj_center['x'] > tail_point[0] and obj_center['y'] <= tail_point[1]:
        tanx = abs(obj_center['y'] - tail_point[1]) / abs(obj_center['x'] - tail_point[0])
        deg = 180 + np.arctan(tanx)*57.2958

    # the tail point is in 4th-quadrant
    if obj_center['x'] <= tail_point[0] and obj_center['y'] < tail_point[1]:
        if obj_center['x'] < tail_point[0]:
            tanx = abs(obj_center['y'] - tail_point[1]) / abs(obj_center['x'] - tail_point[0])
            deg = 360 - np.arctan(tanx)*57.2958

        # when x coordinate of the obj_center and the tail point are the same
        else:
            deg = 270

    return deg


def find_direction(obj_center, obj_points):
    """classify its direction by putting in one of the bins with a range of angles in degrees
    
    :params 
        - obj_center: dict, the center coordinate of the polygon calculated by using the labeled coordinates of a chicken
        - obj_points: list, the 1st coordinate of the polygon which indicates the head of a chicken
    """

    # range of degrees for each bin
    bin_range = 45
    bin_rotate = int(bin_range // 2)
    
    # calculate angle between center and tail point
    deg = calc_angle(obj_center, obj_points)

    # apply rotation to put it into the bin more conveniently
    # east: 338~22 - 0~44, south: 248~292 - 270~314, southeast: 293~337 - 315~359
    rot_deg = deg + bin_rotate

    return rot_deg


def check_CCW(obj_points):
    """verify whether the annotations have been correctly retained between the frames by checking annotation files (.json) 
    created from the tool LabelMe and save all the erroneous frames and their error messages in a log file (.txt)
    
    :params 
        - obj_points: np.array(8,2), labeled coordinates of the polygon of a chicken
    """

    sum = 0

    for i in range(len(obj_points)):
        if i == len(obj_points)-1:
            p1 = obj_points[i]
            p2 = obj_points[0]

        else:
            p1 = obj_points[i]
            p2 = obj_points[i+1]
        
        sum += (p2[0] - p1[0])*(p2[1] + p1[1])

    return True if sum > 0 else False


def extract_labelme(json_file, out_file, error_set):
    """extracts only the relevant information in the annotation files(.json) created from LabelMe

    :params 
        - json_file: str, path where you have your .json file 
        - out_file:
        - error_set:
    """

    with open(json_file, 'r') as _file:
        data = json.load(_file) # return JSON file object as dict

        # only extract relevant information
        filename = data["imagePath"]
        width = data["imageWidth"]
        height = data["imageHeight"]
        obj_list = data['shapes'] # list of dicts

        # check if the imagePath in this json file matches the name of the json file
        file_num = int(filename.split('/')[-1].split(".")[0]) 
        json_num = int(json_file.split('/')[-1].split(".")[0]) 

        if file_num != json_num or len(filename.split('/')) > 1:
            # log error in the text file
            out_file.write('{}: Incorrect ImagePath in json file!\n'.format(json_num))
            error_set.add(json_num)

            # Read from json file and replace imagepath with correct one
            with open(json_file, 'r') as openfile:
                json_object = json.load(openfile)
                filename = "{}.jpg".format("0"*(8-len(str(json_num))) + str(json_num))
                json_object["imagePath"] = filename

            # save it to json file
            with open(json_file, 'w') as f:
                json.dump(json_object, f, indent=2)

    return filename, width, height, obj_list


def sanitize(json_dir, out_path=None):
    """verify whether the annotation guidelines based on SOP have been correctly retained between the frames in 
    annotation files (.json) created from LabelMe and save all the erroneous frames and error messages in a text file
    
    :param 
        - json_dir: str, path where you have your .json files for a video
        - out_path: str, path where you want to store your error text files
    """  

    flag = False
    check = False
    error_set = set([])
    th = 200
    # bin = {0:'east', 1:'northeast', 2:'north', 3:'northwest', 4:'west', 5:'southwest', 6:'south', 7:'southeast'}
    video_num = json_dir.split('/')[-1]
    labelFiles = sorted(glob.glob(os.path.join(json_dir, '*.json'))) 
    imgFiles = sorted(glob.glob(os.path.join(json_dir, '*.jpg')))

    # create a log file for logging errors
    if out_path is None:
        out_file = open(os.path.join('/Users/wonsmacbook/Google_Drive/datasets/chicken/data/', video_num, "{}_sanity.txt".format(video_num)), "w") 
        # out_file2 = open(video_num + "_directions.txt", "a") # append mode
    else:
        out_file = open(os.path.join(out_path, "{}_sanity.txt".format(video_num)), "w")
        # out_file2 = open(os.path.join(out_path, str(video_num) + "_directions.txt", "a")) # append mode
        
    # [1] check if you have the same jpeg and json files in order and extract a list of the json files from the folder and sort them by frame number
    for img, label in zip(imgFiles, labelFiles):
        if int(img.split('/')[-1].split('.')[0]) != int(label.split('/')[-1].split('.')[0]):
            check = True
            
    if check is True:
        out_file.write("The number and order of image and json files are not the same!\n")
    
    # for each json file: /Users/wonsmacbook/Google_Drive/datasets/chicken/data/244/00000390.json
    for json_file in labelFiles:

        # [2] check if the info inside json file is correct and extract filename, coordinates and id info of chickens
        filename, _, _, objs = extract_labelme(json_file, out_file, error_set) # list of jsons (obj = json)
        filename = filename.split('.')[0]
        '''intra-frame'''
        # rearrange information of chickens in the current frame in dict(id:coodinates)
        current_obj = {obj["label"]:np.array(obj["points"]) for obj in objs} # for each json

        for key in sorted(current_obj.keys()):

            # [2] check if the number of coordinates for this chicken is correct
            if len(current_obj[key]) not in [6,8]:
                out_file.write('{}: # of coordinates in chicken {} is {}\n'.format(filename, key, len(current_obj[key])))
                error_set.add(filename)
                continue

            # [3] check if the order of the coordinates for this chicken is counter-clock wise
            if len(current_obj[key]) == 8 and check_CCW(current_obj[key]) == False:
                out_file.write('{}: the order of the coordinates in chicken {} is clock-wise\n'.format(filename, key))
                error_set.add(filename)

        '''inter-frame'''
        # use flag to begin comparison from the second frame (skips this in the first frame)
        if flag is True:  

            # for each chicken in the previous frame
            for key in sorted(prev_obj.keys()):
                
                # [4] check if the same chicken from the previous frame remains in the current one
                if not key in current_obj.keys():
                    out_file.write('{}: chicken {} is missing\n'.format(filename, key))
                    error_set.add(filename)
                    continue
                """
                # [5] find the direction of the chicken's body
                if (len(prev_obj[key]) == 8 or len(prev_obj[key]) == 6) and (len(current_obj[key]) == 8 or len(current_obj[key]) == 6):
                
                    # get the center point of the chicken
                    current_center = get_center(current_obj[key])
                    prev_center = get_center(prev_obj[key])

                    # find the angle between the center of the polygon and the tail and put it in one of the 8 bins
                    current_direct = find_direction(current_center, current_obj[key]) 
                    prev_direct = find_direction(prev_center, prev_obj[key]) 

                    if abs(current_direct - prev_direct) >= 90:
                        current_direct = current_direct % 360
                        prev_direct = prev_direct % 360 
                        out_file2.write('{}: the direction of chicken {}`s tail changed from {} to {}\n'.format(filename, key, bin[prev_direct//45], bin[current_direct//45]))
                """
                # [5] compare the coordinate values that their difference is within a certain threshold
                if len(prev_obj[key]) == len(current_obj[key]):
                    for i in range(len(prev_obj[key])):
                        if abs(prev_obj[key][i][0] - current_obj[key][i][0]) > th:
                            out_file.write('{}: x coordinate {} for chicken {} has changed more than {} pixels\n'.format(filename, i+1, key, th))
                            error_set.add(filename)

                        if abs(prev_obj[key][i][1] - current_obj[key][i][1]) > th:
                            out_file.write('{}: y coordinate {} for chicken {} has changed more than {} pixels\n'.format(filename, i+1, key, th))
                            error_set.add(filename)
                    
        prev_obj = current_obj
        
        if flag is False:
            flag = True
    
    out_file.write("\nFrames that need to be checked: " + str(error_set))
    out_file.close()
    # out_file2.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser() # create an ArgumentParser object
    parser.add_argument("in_path", help="data directory path where you have .json files created from LabelMe")
    parser.add_argument("out_path", nargs='?', default=None, help="dir path where you want to save output files")
    parser.add_argument("-c", "--coco", action="store_true", help="save json file in coco format")
    parser.add_argument("-y", "--yolo", action="store_true", help="save csv and txt file in a YOLO format")
    args = parser.parse_args() # parse_args method inputs a list of strings and outputs a namespace

    # detect and sanitize all json files for a video
    sanitize(args.in_path, args.out_path) 
    print("SANITIZATION FINISHED")

    if args.coco or args.yolo:
        if args.coco:
            labelme_to_coco(args.in_path, args.out_path)

        if args.yolo:
            labelme_to_yolo(args.in_path, args.out_path)

    else:
        print("Please enter one of the optional arguments : -c, -y")