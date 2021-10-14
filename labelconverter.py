"""
labelconverter.py
"""
import numpy as np
import pandas as pd
import glob, json, sys, os, argparse

def labelme_to_coco(json_dir, out_path=None): # annotations, images, 
    """Takes in the directory path containing annotation files in json created from the tool LabelMe  
    and creates json annotation files in COCO format 

    :param json_dir: The path to the folder where you want to load your .json files
    :type json_dir: str

    :param out_path: The path to the folder where you want to save your .csv files
    :type out_path: str
    """    

    coco_dict = {} # dict with 5 keys (info, licenses, images, categories, annotations)
    cat_list = {}
    # list of dicts for required sections for object detection/instance segmentation

    img_list = [] 
    anno_list = [] 
    
    anno_idx = 1 # unique index for each object annotation
    dirFiles = sorted(glob.glob(os.path.join(json_dir, '*.json')))
    img_id = 1
    
    for json_file in dirFiles:
        filename, width, height, obj_list = extract_labelme(json_file)   

        # annotation for an image
        img_record = {
            'id': img_id,
            'width': width,
            'height': height,
            'file_name': filename
        }
        img_id += 1
        img_list.append(img_record)

        if img_record["id"] % 100 == 0:
            print(filename)

        # rearrange list[dict] into dict (key: int / value: (8,2))
        obj_list = {obj["label"]:np.array(obj["points"]) for obj in obj_list} 
        
        # add annotations for each instance in an image
        for key, value in obj_list.items():
            box_dim = get_box_dim(value)
            anno_record = {
                'id': anno_idx,
                'image_id': img_record['id'],
                'category_id': int(key)-1 ,
                'area': box_dim['width'] * box_dim['height'],
                'bbox': [np.min(value[:,0]), np.min(value[:,1]), box_dim['width'], box_dim['height']],
                'is_crowd': 0,
            }
            anno_idx += 1

            polygons = []
            for i in range(value.shape[0]):
                polygons.append([value[i][0], value[i][1]])
            anno_record['segmentation'] = polygons

            anno_list.append(anno_record)

    coco_dict["images"] = img_list
    coco_dict["categories"] = cat_list
    coco_dict["annotations"] = anno_list

    if out_path is None:
        with open('instances_{}_{}.json'.format(json_dir.split("/")[-2],json_dir.split("/")[-1]), 'w') as fp:
            json.dump(coco_dict, fp,  indent=4)

    else:
        with open(os.path.join(out_path,'labels_coco.json'), 'w') as fp:
            json.dump(coco_dict, fp,  indent=4)

    print("CONVERTED TO COCO FORMAT!")

    return coco_dict


def labelme_to_yolo(json_dir, out_path=None):
    """load annotation files(.json) created from tool LabelMe and creates txt files for each image in YOLO format 
    (the file should have the following columns in order: object-id, x, y, width, height)

    :params 
        - json_dir: str, path to the folder where your .json files are
        - out_path: str, path to the folder where you want to save your .csv and .txt files
    """    

    # Loop through every json file for every image    
    dirFiles = sorted(glob.glob(os.path.join(json_dir, '*.json')))

    for json_file in dirFiles:
        # extract relevant information in each .json file
        filename, img_width, img_height, obj_list = extract_labelme(json_file)

        json_num = int(json_file.split('/')[-1].split(".")[0])
        file_num = int(filename.split(".")[0])

        if out_path is None:
            out_file = open(os.path.join(json_dir, filename.split(".")[0] + ".txt"), "w")

        else:
            out_file = open(os.path.join(json_dir, filename.split(".")[0] + ".txt"), "w")
        
        # Each object represents each actual image label
        for obj in obj_list: # obj = dict
            # label = obj["label"] # str
            label = 0
            points = np.array(obj["points"]) # list of lists (8 points of [x,y])

            # calculate center, width and height
            center = get_center(points)
            box = get_box_dim(points)
            
            # image of size 944 x 1080 = height 1080, width 944
            # normalize center, width and height by image width and height
            center['x'] = center['x'] / img_width
            center['y'] = center['y'] / img_height
            box['width'] = box['width'] / img_width
            box['height'] = box['height'] / img_height

            # save into txt file for each image: write each row in (object-id center_x center_y width height) format
            record = " ".join([str(label), str(center['x']), str(center['y']), str(box['width']), str(box['height'])]) + '\n'
            out_file.write(record) 
        out_file.close() 

    print("CONVERTED TO YOLO FORMAT!")


def get_box_dim(obj_points):
    """Find the width and height of the box bounding a chicken
    
    :param obj_points: labeled coordinates of the polygon of a chicken
    :type obj_points: numpy array of size (8,2)
    """  
    width = np.max(obj_points[:,0]) - np.min(obj_points[:,0])
    height = np.max(obj_points[:,1]) - np.min(obj_points[:,1])
    return {'width':width, 'height':height}


def get_center(obj_points):
    """calculates the center point of the polygon by using the labeled coordinates of a chicken
    needed for YOLO or COCO

    :param obj_points: labeled coordinates of the polygon of a chicken
    :type obj_points: numpy array of size (8,2)
    """  
    x = (np.min(obj_points[:,0]) + np.max(obj_points[:,0])) / 2.0
    y = (np.min(obj_points[:,1]) + np.max(obj_points[:,1])) / 2.0

    return {'x':x,'y':y}


def extract_labelme(json_file):
    """Extracts only the relevant information in the annotation files(.json) created from LabelMe  

    :param json_dir: The path to the folder where your .json files are
    :type json_dir: str
    """

    f = open(json_file, "r") # open JSON file
    data = json.load(f) # return JSON file object as dict

    # extract only meaningful information for all objects in an image
    filename = data["imagePath"].split('/')[-1]
    width = int(data["imageWidth"])
    height = int(data["imageHeight"])
    obj_list = data['shapes'] # list of dicts

    return filename, width, height, obj_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser() # create an ArgumentParser object
    parser.add_argument("in_path", help="path where you have .json files created from LabelMe")
    parser.add_argument("out_path", nargs='?', default=None, help="path where you want to save output files")
    parser.add_argument("-c", "--coco", action="store_true", help="save json file in COCO format")
    parser.add_argument("-y", "--yolo", action="store_true", help="save txt file in YOLO format")
    args = parser.parse_args() # parse_args method inputs a list of strings and outputs a namespace

    if args.coco or args.yolo:
        if args.coco:
            labelme_to_coco(args.in_path, args.out_path)

        if args.yolo:
            labelme_to_yolo(args.in_path, args.out_path)

    else:
        print("Please enter one of the optional arguments : -c, -y")