import pandas as pd

###########################################################################################################
###########################################################################################################
###########################################################################################################

#Used to verify PKL file from DataProcessing is made correctly
#Please run and make sure there is a corresponding RGB jpg for each Depth Png
#Check both the beginning and the end of the file for any shifts
# EX. 'frame_700000.jpg': 'frame_700000.png'

#If there are any that dont match numerically EX.'frame_700010.jpg': 'frame_700000.png' then there is something wrong
#with the dataset and you should fix it before Training the model.


###########################################################################################################
###########################################################################################################
###########################################################################################################

df2 = pd.read_pickle(r'C:\Users\Admin\Downloads\pytorch_ipynb\DataSet\index1.pkl')
# print the dataframe
print(df2)

df2 = pd.read_pickle(r'C:\Users\Admin\Downloads\pytorch_ipynb\DataSet\index2.pkl')
# print the dataframe
print(df2)