import argparse
import open3d as o3d
import os
import json
import sys
from PIL import Image

pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(pwd, '..'))
from initialize_config import initialize_config

innie = r'C:\Users\benhu\Desktop\Test_Dataset' # where the file goes
outie = r'C:\Users\benhu\Desktop\mkv_file\ecj_1202.mkv'## input file

def squareimage(image):
    # Open the image
    im = Image.open(image)

    # Get the aspect ratio
    aspect_ratio = im.width / im.height

    # Calculate the size of the square
    size = 572

    # Crop the image
    if aspect_ratio > 1:
        # Landscape image
        left = (im.width - im.height) / 2
        right = left + im.height
        top = 0
        bottom = im.height
    else:
        # Portrait image
        left = 0
        right = im.width
        top = (im.height - im.width) / 2
        bottom = top + im.width

    im = im.crop((left, top, right, bottom))

    # Resize the image to the square shape
    im = im.resize((size, size))

    # Save the image
    im.save(image)

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
    
    folder_name = cwd.rsplit('\\', 1)[1]
    print(" \n --- '{}' folder has been EXITED --- \n".format(folder_name))

class ReaderWithCallback:

    def __init__(self, input, output):
        self.flag_exit = False
        self.flag_play = True
        self.input = input
        self.output = output

        self.reader = o3d.io.AzureKinectMKVReader()
        self.reader.open(self.input)
        if not self.reader.is_opened():
            raise RuntimeError("Unable to open file {}".format(outie))

    def escape_callback(self, vis):
        self.flag_exit = True
        return False

    def space_callback(self, vis):
        if self.flag_play:
            print('Playback paused, press [SPACE] to continue.')
        else:
            print('Playback resumed, press [SPACE] to pause.')
        self.flag_play = not self.flag_play
        return False

    def run(self):
        glfw_key_escape = 256
        glfw_key_space = 32
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, self.escape_callback)
        vis.register_key_callback(glfw_key_space, self.space_callback)

        vis_geometry_added = False
        vis.create_window('reader', 1920, 540)

        print(
            "MKV reader initialized. Press [SPACE] to pause/start, [ESC] to exit."
        )

        if self.output is not None:
            abspath = os.path.abspath(self.output)
            metadata = self.reader.get_metadata()
            o3d.io.write_azure_kinect_mkv_metadata(
                '{}/intrinsic.json'.format(abspath), metadata)

            config = {
                'path_dataset': abspath,
                'path_intrinsic': '{}/intrinsic.json'.format(abspath)
            }
            initialize_config(config)
            with open('{}/config.json'.format(abspath), 'w') as f:
                json.dump(config, f, indent=4)

        idx = 0
        while not self.reader.is_eof() and not self.flag_exit:
            if self.flag_play:
                rgbd = self.reader.next_frame()
                if rgbd is None:
                    continue

                if not vis_geometry_added:
                    vis.add_geometry(rgbd)
                    vis_geometry_added = True

                if self.output is not None:
                    color_filename = '{0}/RGB_Images/frame_{1:05d}.jpg'.format(
                        self.output, idx)
                    cwd = os.getcwd()
                    
                    #print('Writing to {}'.format(color_filename))
                    o3d.io.write_image(color_filename, rgbd.color)
                    #enter_folder("RGB_Images")
                    squareimage(color_filename)
                    #exit_folder()

                    depth_filename = '{0}/Depth_Maps/frame_{1:05d}.png'.format(
                        self.output, idx)
                    
                    #print('Writing to {}'.format(depth_filename))
                    o3d.io.write_image(depth_filename, rgbd.depth)
                    #enter_folder("Depth_Maps")
                    squareimage(depth_filename)
                    #exit_folder()
                    idx += 1

            try:
                vis.update_geometry(rgbd)
            except NameError:
                pass
            vis.poll_events()
            vis.update_renderer()

        self.reader.close()


if __name__ == '__main__':

    if innie is None:
        print('No output path, only play mkv')
    elif os.path.isdir(innie):
        print('Output path {} already existing, only play mkv'.format(
            innie))
        innie = None
    else:
        try:
            os.mkdir(innie)
            os.mkdir('{}/color'.format(innie))
            os.mkdir('{}/depth'.format(innie))
        except (PermissionError, FileExistsError):
            print('Unable to mkdir {}, only play mkv'.format(innie))
            innie = None

    reader = ReaderWithCallback(outie, innie)
    reader.run()
