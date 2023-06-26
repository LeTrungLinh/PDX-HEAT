# import cv2
# import os
# import scipy.io as sio
# import numpy as np 

# images = './data_train/chituyen_6_6/rgb/'

# labels = './data_train/chituyen_6_6/annot/'

# extentions = ('.jpg','.png','.PNG','.jpeg')
# for filename in os.listdir(images):
#     # the end of image file is .jpg or png 

#     if filename.endswith(extentions):
#         # head = filename.split(extentions)[0]
#         head = filename.split('.')[0]
#         print(head)
#         # img = cv2.imread(images +head+ '.jpg')
#         lines = sio.loadmat(images +head+ '.mat')['lines']
#         points_new = {}
#         for p1, p2 in lines:
#             p1 = tuple(p1)
#             p2 = tuple(p2)
#             if p1 not in points_new:
#                 points_new[p1] = []
#             if p2 not in points_new:
#                 points_new[p2] = []
#             points_new[p1].append(list(p2))
#             points_new[p2].append(list(p1))
#         # save points_new to npy file 
#         np.save(labels+ head +'.npy', points_new)






############################## Create txt file for train ############################################3
import random
import os 


base = 'data_train'
extentions = ('.jpg','.png','.PNG','.jpeg', 'JPG')
# list all file '.jpg','.png ' in the directory
for (root, dirs, files) in os.walk(base,topdown= True):
    for name in files:
        if name.endswith(extentions):
            filename = os.path.join(root, name)
            with open(base + '/' + 'full_list.txt', 'a') as f:
                f.writelines(filename+'\n')

# Read the text file containing the paths to image files
with open(base +'/' + 'full_list.txt') as f:
    file_paths = f.readlines()

# Shuffle the list for randomization
random.shuffle(file_paths)

# Determine the lengths of the two files
total_items = len(file_paths)
train_len = int(total_items * 0.8)
test_valid_len = total_items - train_len
test_len = int(test_valid_len * 0.5)
valid_len = test_valid_len - test_len

# Split the list into three files
with open(base + '/' + "train_list.txt", "w") as train_file:
    train_file.writelines(file_paths[:train_len])

with open(base + '/' + "test_list.txt", "w") as test_file:
    test_file.writelines(file_paths[train_len:train_len + test_len])

with open(base + '/' + "valid_list.txt", "w") as valid_file:
    valid_file.writelines(file_paths[train_len + test_len:])


