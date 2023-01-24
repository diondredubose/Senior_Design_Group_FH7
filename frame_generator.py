# Created by: Diondre Dubose (01/23/23)
# Edited by: 
#  The code converts a set of .mkv files into frames

# Import dependecies
import os
from zipfile import ZipFile #unzips zip files
import cv2


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
    print(" \n --- '%s' folder has been ENTERED --- \n",folder_name)

def exit_folder():
    #gets current working directory (cwd)
    cwd = os.getcwd()

    #path of parent directory (pd)
    # os.path.dirname('/home/user/Documents/my_folder') would return /home/user/Documents
    pd_path = os.path.dirname(cwd)

    # go to directory specified by pd_path
    os.chdir(pd_path)
    
    folder_name = cwd.rsplit('/',1)[1]
    print(" \n --- '%s' folder has been EXITED --- \n", folder_name)

def create_folder(folder_name):
    #gets current working directory (cwd)
    cwd = os.getcwd()

    folder_path = os.path.join(cwd, folder_name)
    os.mkdir(folder_path)
    print(" \n --- '%s' folder has been CREATED --- \n", folder_name)

def extract_zip(file_name):
  with ZipFile(file_name, 'r') as zip:
    print("\n Extracting '%s' ... \n", file_name)
    zip.extractall()
    print("\n --- Extraction of %s Complete --- \n", file_name)

def extract_zip_in_folder(file_name, folder_name):
  with ZipFile(file_name, 'r') as zip:
    print("\n Extracting '%s' ... \n", file_name)
    create_folder(folder_name)
    enter_folder(folder_name)
    zip.extractall()
    exit_folder()
    print("\n --- Extraction of %s Complete --- \n", file_name)

def video_expander(video_file):
    # Extract the RGB image and depth map from each frame of a .mkv video file
    video = cv2.VideoCapture(video_file)

    # Number of frames in video
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    folder_name = video_file.rsplit('.',1)[0]
    create_folder(folder_name)
    enter_folder(folder_name)

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
            rgb_image = frame

            # Extract the depth map (assuming the video has a depth map embedded)
            depth_map = frame[:, :, 2]

            # Do something with the RGB image and depth map
            """
            By default, the cv2.imwrite() function will save the JPG file to the 
            current working directory, which is the directory where the script is 
            located.
            """
            # Convert frame (numpy ndarray) to image
            cv2.imwrite("frame_rgb_{}.jpg".format(i), rgb_image)
            cv2.imwrite("frame_depth_{}.jpg".format(i), depth_map)

        else:
            # Handle the case where the frame could not be read
            print("Frame #%f was not read successfully", i)
            
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

    extract_zip_in_folder(zip_file,folder_name)
    enter_folder(folder_name)
    
    cwd = os.getcwd()
    for video_file in os.listdr(cwd):
        video_expander(video_file)

    exit_folder()

frame_generator("sample.mkv.zip")









