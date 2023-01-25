# Created by: Diondre Dubose (01/23/23)
# Edited by: 
#  The code converts a set of .mkv files into frames 

# Import dependecies
import os
from zipfile import ZipFile #unzips zip files
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

"""
Ability 1: Enter folder
Ability 2: Exit folder
Ability 3: Create folder
Ability 4: Extract .zip files
Ability 5: Extract the RGB image 
           and depth map from each
           frame of a .mkv video file
"""


def enter_folder(folder_name):
    #gets current working directory (cwd)
    cwd = os.getcwd()

    #create folder path
    folder_path = os.path.join(cwd, folder_name)

    # go to directory specified by folder path
    os.chdir(folder_path)

    #prints the name of folder that is entered
    print(" \n --- '{}' folder has been ENTERED --- \n".format(folder_name))

def exit_folder():
    #gets current working directory (cwd)
    cwd = os.getcwd()

    #path of parent directory (pd)
    # os.path.dirname('/home/user/Documents/my_folder') would return /home/user/Documents
    pd_path = os.path.dirname(cwd)

    # go to directory specified by pd_path
    os.chdir(pd_path)
    
    folder_name = cwd.rsplit('/',1)[1]
    print(" \n --- '{}' folder has been EXITED --- \n".format(folder_name))

def create_folder(folder_name):
    #gets current working directory (cwd)
    cwd = os.getcwd()

    folder_path = os.path.join(cwd, folder_name)
    if os.path.isdir(folder_path):
        print("\n --- '{}' folder already exists, skipping creation process --- \n".format(folder_name))
    else:
        os.mkdir(folder_path)
        print(" \n --- '{}' folder has been CREATED --- \n".format(folder_name))


def extract_zip(file_name):
    # check if the folder with the same name as the zip file already exists
    folder_name = file_name.rsplit('.',1)[0]
    if os.path.isdir(folder_name):
        print("\n --- Extraction of '{}' already exists, skipping extraction process --- \n".format(file_name))
        return
    with ZipFile(file_name, 'r') as zip:
        print("\n Extracting '{}' ... \n".format(file_name))
        zip.extractall()
        print("\n --- Extraction of '{}' Complete --- \n".format(file_name))


def extract_zip_in_folder(file_name, folder_name):
    if os.path.isdir(folder_name):
        print("\n --- Extraction of '{}' already exists, skipping extraction process --- \n".format(file_name))
        return
    with ZipFile(file_name, 'r') as zip:
        print("\n Extracting '{}' ... \n".format(file_name))
        create_folder(folder_name)
        enter_folder(folder_name)
        zip.extractall()
        exit_folder()
        print("\n --- Extraction of '{}' Complete --- \n".format(file_name))


def video_expander(video_file):
    # Extract the RGB image and depth map from each frame of a .mkv video file
    video = video = cv2.VideoCapture(video_file, cv2.CAP_OPENNI_DEPTH_MAP)

    # Number of frames in video
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    folder_name = video_file.rsplit('.',1)[0]
    create_folder(folder_name)
    enter_folder(folder_name)

    create_folder("RGB_Images")
    create_folder("Depth_Maps")
    for i in range(num_frames):
    # Read the next frame
        valid, frame = video.read()
        """
        In the code, ' valid, frame = video.read() ' reads the next frame 
        from the video and assigns it to the frame variable. The ret variable 
        is a Boolean that is true if the frame was read successfully, otherwise 
        it is false. The type of frame is a numpy ndarray.
        """
         
        # Check if the frame was successfully read
        if valid:
            # Extract the RGB image
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extract the depth map (assuming the video has a depth map embedded)
            depth_map = frame[:,:,1]

            # Do something with the RGB image and depth map
            """
            By default, the cv2.imwrite() function will save the JPG file to the 
            current working directory, which is the directory where the script is 
            located.
            """
            # Convert frame (numpy ndarray) to image
            enter_folder("RGB_Images")
            cv2.imwrite("frame_{}.jpg".format(i), rgb_image)
            exit_folder()

            enter_folder("Depth_Maps")
            cv2.imwrite("frame_{}.jpg".format(i), depth_map)
            exit_folder()

        else:
            # Handle the case where the frame could not be read
            print("Frame #{} was not read successfully".format(i))
            
    exit_folder()
    # Release the video capture object
    video.release()

def frame_generator(zip_file):
    folder_name = zip_file.rsplit('.',1)[0]   
    while(True):
        if '.' in folder_name:
            folder_name = folder_name.rsplit('.',1)[0]
        else:
            break

    extract_zip(zip_file)
    enter_folder(folder_name)
    
    cwd = os.getcwd()
    FILES = [file for file in os.listdir(cwd) if file.endswith(".mkv")]
    for video_file in FILES:
        if not os.path.exists(os.path.join(cwd, video_file.rsplit('.',1)[0])):
            video_expander(video_file)
        else:
            print("{} has already been extracted".format(video_file))
    exit_folder()

class AgentDataset(Dataset):
    def __init__(self, zip_file):
        frame_generator(zip_file)
        folder_name = zip_file.rsplit('.',1)[0]   
        while(True):
            if '.' in folder_name:
                folder_name = folder_name.rsplit('.',1)[0]
            else:
                break
        cwd = os.getcwd()
        cwd = os.path.join(cwd, folder_name)
        self.array = []
        oldcwd = cwd
        for file in os.listdir(cwd):
            if file.endswith(".mkv"):
                i = file.rsplit('.',1)[0]   
                while(True):
                    if '.' in i:
                        i = i.rsplit('.',1)[0]
                    else:
                        break
                self.array.append(i)
        self.rgb_images = [] 
        self.depth_maps = []
        for folder in self.array:  
            cwd = oldcwd
            cwd = os.path.join(cwd, folder)
            self.x = [os.path.join(cwd,"RGB_Images", file) for file in os.listdir(os.path.join(cwd, "RGB_Images")) if file.endswith(".jpg")]
            self.y = [os.path.join(cwd, "Depth_Maps", file) for file in os.listdir(os.path.join(cwd, "Depth_Maps")) if file.endswith(".jpg")]
            self.rgb_images += self.x
            self.depth_maps += self.y

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_img = cv2.imread(self.rgb_images[idx])
        depth_map = cv2.imread(self.depth_maps[idx])
        return rgb_img, depth_map

ECJ = AgentDataset("roomrecordings_2023_01_22.zip")
num_frames = ECJ.__len__()
for frame in range(ECJ.__len__()):    
    image, depth = ECJ.__getitem__(frame)
#dataloader = torch.utils.data.DataLoader(ECJ, batch_size=32, shuffle=True)
