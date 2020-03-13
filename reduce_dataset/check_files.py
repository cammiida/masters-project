import json
import os


def check_json(file):
    with open(file, 'rb') as json_file:
        data = json.load(json_file)
        print("Images: ", data["images"][0])
        print("Annotations: " ,data["annotations"][0])
        print("Categories: ", data["categories"][0])
        for key, value in data.items():
            print("Key: ", key)



if __name__ == '__main__':
    big_root = '../data/big'
    small_root = '../data/small'

    check_json(os.path.join(big_root, 'annotations/person_keypoints_train2014.json'))
