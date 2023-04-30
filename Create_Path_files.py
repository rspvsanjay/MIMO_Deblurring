import os

path1 = "/content/drive/MyDrive/StartCode3/GoPro_Large/train/"
dir_list = os.listdir(path1)
input_train = []
output_train = []
for num1 in range(len(dir_list)):
    path2 = path1 + dir_list[num1] + "/blur/"
    image_list = os.listdir(path2) # read the image files name present in path2
    path3 = path1 + dir_list[num1] + "/sharp/"
    for num2 in range(len(image_list)): 
        path4 = path2 + image_list[num2] # make input path
        path5 = path3 + image_list[num2]  # make output path
        if os.path.exists(path4) and os.path.exists(path5): # check the input and output files are present or not
          path4 = path2 + image_list[num2] + "\n" # go to next line using \n for input path
          path5 = path3 + image_list[num2] + "\n" # go to next line using \n for output path
          input_train.append(path4)
          output_train.append(path5)

save_path = "/content/drive/MyDrive/StartCode3/"
path = save_path + "input_train.txt"
file1 = open(path,"w")
file1.writelines(input_train)
file1.close()

path = save_path + "output_train.txt"
file1 = open(path,"w")
file1.writelines(output_train)
file1.close()
