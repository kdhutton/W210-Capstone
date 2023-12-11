import json
import os
import random
from shutil import move
import s3fs
import json
import getpass
import tarfile
import os
from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv()
    
    access_key = os.getenv('AWS_ACCESS_KEY')
    secret_key = os.getenv('AWS_SECRET_KEY')
    
    
    # Run once to get images onto EC2
    
    wider_dir = './WIDER'
    if not os.path.exists(wider_dir):
        os.makedirs(wider_dir)
    
    # Specify your S3 bucket and file path
    bucket_name = '210bucket'
    s3_file_path = 'wider_attribute_image.tgz'
    
    # Initialize an S3 filesystem
    s3 = s3fs.S3FileSystem(key=access_key, secret=secret_key)
    
    # Download the .tgz file from S3
    with s3.open(f"{bucket_name}/{s3_file_path}", 'rb') as s3_file:
        with tarfile.open(fileobj=s3_file, mode="r:gz") as tar:
            # Specify the destination directory where you want to store the extracted contents
            extract_dir = wider_dir # Change this to your desired directory
            tar.extractall(path=extract_dir)
    
    print("File downloaded and extracted successfully.")
    
    # Specify your S3 bucket and directory path
    s3_directory_path = 'wider_attribute_annotation/'
    
    local_directory = './WIDER/Annotations'  # Change this to your desired directory
    
    s3_files = s3.ls(f"{bucket_name}/{s3_directory_path}")
    
    
    # Create the local directory if it doesn't exist
    os.makedirs(local_directory, exist_ok=True)
    
    # Download each file from the S3 directory to the local directory
    for s3_file in s3_files:
        # Get the filename from the S3 file path
        filename = os.path.basename(s3_file)
        
        # Download the file to the local directory
        local_path = os.path.join(local_directory, filename)
        with s3.open(s3_file, 'rb') as s3_file_obj:
            with open(local_path, 'wb') as local_file:
                local_file.write(s3_file_obj.read())
    
    print("Files downloaded successfully.")
    
    def split_and_move_images(json_data, trainval_json, split_ratio=0.5):
        # Create a dictionary to store images by class
        class_images = {}
        for i in json_data['images']:
            scene_id = i['scene_id']
            file_name = i['file_name']
            image_data = i['targets']
    
            # Add the image to the class_images dictionary
            if scene_id not in class_images:
                class_images[scene_id] = []
            class_images[scene_id].append({'scene_id': scene_id, 'file_name': file_name, 'targets': image_data})
        
        # Create wider_attribute_trainval.json
        trainval_data = []
    
        # Process each class
        for scene_id, images in class_images.items():
            # Shuffle the images in the class
            random.shuffle(images)
    
            # Select half of the images
            split_index = int(len(images) * split_ratio)
            trainval_images = images[:split_index]
            test_images = images[split_index:]
    
            # Add selected images to wider_attribute_trainval.json
            trainval_data.extend(trainval_images)
    
            # Remove sampled images from test_data
            for test_image in test_images:
                json_data['images'].remove({'scene_id': scene_id, 'file_name': test_image['file_name'], 'targets': test_image['targets']})
        
        # Save updated wider_attribute_test.json
        with open("WIDER/Annotations/wider_attribute_test.json", 'w') as test_file:
            json.dump(json_data, test_file)
    
        with open(trainval_json, 'r') as trainval_file:
            existing_data = json.load(trainval_file)
    
        # Update the existing 'images' values with trainval_data
        existing_data['images'].extend(trainval_data)
    
        # Write the updated data back to the file
        with open(trainval_json, 'w') as trainval_file:
            json.dump(existing_data, trainval_file)
    
    if __name__ == "__main__":
        # Load wider_attribute_test.json
        with open("WIDER/Annotations/wider_attribute_test.json", 'r') as test_file:
            test_data = json.load(test_file)
    
        # Define the path for wider_attribute_trainval.json
        trainval_json_path = "WIDER/Annotations/wider_attribute_trainval.json"
    
        # Call the function to split and move images
        split_and_move_images(test_data, trainval_json_path)
    
    def make_wider(tag, data_path):
        img_path = os.path.join(data_path, "Image")
        ann_path = os.path.join(data_path, "Annotations")
        ann_file = os.path.join(ann_path, "wider_attribute_{}.json".format(tag))
    
        data = json.load(open(ann_file, "r"))
    
        image_list = data['images']
        # for image in image_list:
        #     for person in image["targets"]: # iterate over each person
        #         tmp = {}
        #         tmp['img_path'] = os.path.join(img_path, image['file_name'])
        #         tmp['bbox'] = person['bbox']
        #         attr = person["attribute"]
        #         for i, item in enumerate(attr):
        #             if item == -1:
        #                 attr[i] = 0
        #             if item == 0:
        #                 attr[i] = 0  # pad un-specified samples
        #             if item == 1:
        #                 attr[i] = 1
        #         tmp["target"] = attr
        #         final.append(tmp)
    
        json.dump(image_list, open("data/wider/{}_wider.json".format(tag), "w"))
        print("data/wider/{}_wider.json".format(tag))
    
    #run once
    if not os.path.exists("data/wider"):
        os.makedirs("data/wider")
    
    # 0 (zero) means negative, we treat un-specified attribute as negative in the trainval set
    make_wider(tag='trainval', data_path='WIDER') 
    make_wider(tag='test', data_path='WIDER')

