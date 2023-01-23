import os
import cv2


# Open the "nyu_subset" folder in the working directory
nyu_subset_folder = "nyu_subset"
for folder in os.listdir(nyu_subset_folder):
    folder_path = os.path.join(nyu_subset_folder, folder)
    if os.path.isdir(folder_path):
        # Open the "INDEX.txt" file in the current folder
        index_file = os.path.join(folder_path, "INDEX.txt")
        with open(index_file) as f:
            # Create a list to store the file names of the image pairs
            image_pairs = []
            # Parse through the "INDEX.txt" file
            for line in f:
                line = line.strip()
                # Split the line into the file name and file type
                file_name = line
                # Check if the file name is prefixed with "d" or "r"
                if line.endswith(".pgm") == True:
                    depth_file = file_name
                elif line.endswith(".ppm") == True:
                    color_file = file_name
                if 'color_file' in locals() and 'depth_file' in locals():
                    # Append the image pair to the list
                    image_pairs.append((color_file, depth_file))
            # create a text file that contains the list of all pairs of .ppm and .pgm file names
            txt_file = os.path.join(folder_path, "image_pairs.txt")
            with open(txt_file, 'w') as t:
                for i in range(len(image_pairs)):
                    t.write(image_pairs[i][0]+' '+image_pairs[i][1]+'\n')
            # Plot the first 5 image pairs using cv2
            for i in range(5):
                color_image_path = os.path.join(folder_path, image_pairs[i][0])
                depth_image_path = os.path.join(folder_path, image_pairs[i][1])
                if os.path.isfile(color_image_path):
                    color_image = cv2.imread(color_image_path)
                    cv2.imshow("Color Image", color_image)
                    cv2.waitKey(0)
                else:
                    print("{} not found".format(color_image_path))
                if os.path.isfile(depth_image_path):
                    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)
                    cv2.imshow("Depth Image", depth_image)
                    cv2.waitKey(0)
            cv2.destroyAllWindows()
                    