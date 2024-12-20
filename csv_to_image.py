from collections import defaultdict
import csv
from PIL import Image
import numpy as np
import string
import os


csv_File_Path = "./A_Z Handwritten Data.csv"

count = 1
last_digit_Name =  None

image_Folder_Path = "images/"

Alphabet_Mapping_List = list(string.ascii_uppercase)

for alphabet in Alphabet_Mapping_List:
    path = os.path.join(image_Folder_Path , alphabet)
    if not os.path.exists(path):
        os.makedirs(path)



dict = defaultdict(int)
finished = 0
with open(csv_File_Path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    count = 0
    for row in reader:
        digit_Name = row.pop(0)
        image_array = np.asarray(row)
        image_array = image_array.reshape(28, 28)
        new_image = Image.fromarray(image_array.astype('uint8'))

        if last_digit_Name != str(Alphabet_Mapping_List[(int)(digit_Name)]):
            last_digit_Name = str(Alphabet_Mapping_List[(int)(digit_Name)])
            count = 0
            print ("")
            print ("Prcessing Alphabet - " + str (last_digit_Name))
        
        
        if dict[digit_Name] == 10:
            finished += 1
            dict[digit_Name] += 1
            if finished >= 26:
                break
            continue
          
        dict[digit_Name] += 1
        if dict[digit_Name] >= 10:
          continue
        
        image_Path = os.path.join(image_Folder_Path, str(last_digit_Name) , f"{ str(last_digit_Name)}-{str(count)}.png")
        # image_Path =  image_Folder_Path + '\\' + last_digit_Name + '\\' + str(last_digit_Name) + '-' + str(count) + '.png'
        new_image.save(image_Path)
        count = count + 1
        
        if count % 1000 == 0:
            print ("Images processed: " + str(count))