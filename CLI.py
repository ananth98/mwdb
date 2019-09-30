import os
from PIL import Image
import cv2
from skimage.feature import local_binary_pattern
import numpy as np
import csv
import json
import shutil
import scipy
import math
import scipy.stats
from skimage.feature import hog

# Paths Given for image files are stored and creation of csv file etc.

directory = 'C:\\Users\\sai\\Desktop\\1-1\\MWDB\\Test_Dataset\\'
csv_path = 'C:\\Users\\sai\\Desktop\\1-1\\MWDB\\Project\\'

# Divide the image into 100*100 resolution , convert it to YUV model and then store

def CM_image_windows(CM_image_store_path,directory,imagename, height, width):
    im = Image.open(directory+imagename)
    imgwidth, imgheight = im.size
    window_number=1
    imgUMat = cv2.imread(directory+imagename)
    img_yuv = cv2.cvtColor(imgUMat, cv2.COLOR_BGR2YUV)
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            crop_img = img_yuv[i:i + width, j:j + width]
            final_img_path=os.path.join(CM_image_store_path,imagename[:-4])
            if not os.path.exists(final_img_path):
                os.makedirs(final_img_path)
            cv2.imwrite(final_img_path+"/"+str(window_number)+".jpg", crop_img)
            window_number +=1

# Performs task1 for Color Moments individually

def CM_task1(CM_image_store_path,imagename):
    im = Image.open(directory+imagename)
    imgwidth, imgheight = im.size
    window_number=1
    for i in range(0,imgheight,100):
        for j in range(0,imgwidth,100):
            box = (j, i, j+100, i+100)
            a = im.crop(box)
            final_img_path=os.path.join(CM_image_store_path,imagename[:-4])
            if not os.path.exists(final_img_path):
                os.makedirs(final_img_path)
            a.save(final_img_path+"/"+str(window_number)+".jpg")
            window_number +=1
    matrix = []
    i = 1
    filelist = [f for f in os.listdir(CM_image_store_path + "\\" + imagename[:-4]) if f.endswith(".jpg")]
    for file in filelist:
        imgUMat = cv2.imread(os.path.join(CM_image_store_path + "\\" + imagename[:-4], file))
        a = np.array(np.mean(imgUMat, axis=(0, 1)))
        b = np.array(np.var(imgUMat, axis=(0, 1)))
        c = np.array(scipy.stats.skew(imgUMat.reshape(-1, 3)))
        d = np.concatenate((np.concatenate((a, b), axis=0), c), axis=0)
        matrix.extend(d.tolist())
        i += 1
    print(matrix)
    shutil.rmtree(final_img_path)

# extract the features for converted images and store it the imagename and the array in a csv file as 2 columns.

def color_moments(CM_image_store_path,imagename):
    matrix = []
    i=1
    filelist = [f for f in os.listdir(CM_image_store_path+"\\"+imagename[:-4]) if f.endswith(".jpg")]
    for file in filelist:
        imgUMat = cv2.imread(os.path.join(CM_image_store_path+"\\"+imagename[:-4] , file))
        a = np.array(np.mean(imgUMat, axis=(0, 1)))
        b = np.array(np.var(imgUMat, axis=(0, 1)))
        c = np.array(scipy.stats.skew(imgUMat.reshape(-1, 3)))
        d = np.concatenate((np.concatenate((a, b), axis=0), c), axis=0)
        matrix.extend(d.tolist())
        i += 1
    with open('Color_Moments.csv', 'a',newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=',')
        writer.writerow([imagename,matrix])

# Perform task2 for color moments as an individual task

def CM_task2(CM_image_store_path):
    if (os.path.exists('Color_Moments.csv')):
        os.remove('Color_Moments.csv')
    for imagename in os.listdir(directory):
        CM_image_windows(CM_image_store_path, directory, imagename, 100, 100)
        color_moments(CM_image_store_path, imagename)


# Find out similar images according to Euclidean Distance and show it.

def CM_similar(imageid,k):
    # creating a dictionary to store the image name and the similarity value

    distance = {}
    with open(csv_path+'Color_Moments.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if (imageid == row[0]):
                cm = json.loads(row[1])
    with open(csv_path+'Color_Moments.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            dist = (sum([(a - b) ** 2 for a, b in zip(cm, json.loads(row[1]))]))**0.5
            distance.update({row[0]: dist})
    sorted_x = sorted(distance.items(), key=lambda x: x[1])
    # storing image name and similarity value in 2 lists
    img = [lis[0] for lis in sorted_x[:k]]
    dist = [lis[1] for lis in sorted_x[:k]]
    for i in range (0,k):
        im = Image.open(directory + img[i])
        im.show()
        print(img[i])
        print(dist[i])

# Perform task3 individually for color moments individually

def CM_task3(CM_image_store_path,imageid,k):
    if (os.path.exists('Color_Moments.csv')):
        os.remove('Color_Moments.csv')
    for imagename in os.listdir(directory):
        CM_image_windows(CM_image_store_path, directory, imagename, 100, 100)
        color_moments(CM_image_store_path, imagename)
    CM_similar(imageid, k)

# Divide the image into 100*100 resolution , convert it to Gray model and then store

def LBP_image_windows(img_store_path,directory, imagename, height, width):
    im = Image.open(directory+imagename).convert('L')
    imgwidth, imgheight = im.size
    window_number=1
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            final_img_path=os.path.join(img_store_path,imagename[:-4])
            if not os.path.exists(final_img_path):
                os.makedirs(final_img_path)
            a.save(final_img_path+"/"+str(window_number)+".jpg")
            window_number +=1

# extract the features for converted images and store it the imagename and the array in a csv file as 2 columns.

def LBP(image_store_path,imagename):
    radius=1
    num_points=8*radius
    lbp_hist_list=[]
    i=1
    filelist = [f for f in os.listdir(image_store_path+"\\"+imagename[:-4]) if f.endswith(".jpg")]
    for file in filelist:
        img = Image.open(os.path.join(image_store_path+"\\"+imagename[:-4] , file))
        lbp = local_binary_pattern(img,num_points,radius,method='uniform')
        (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, num_points + 3),range=(0, num_points + 2))
        lbp_hist_list.extend(hist)
    with open('LBPFeature.csv' , 'a',newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=',')
        writer.writerow([imagename, lbp_hist_list])

# Performs task1 for LBP individually

def LBP_task1(image_store_path,imagename):
    im = Image.open(directory+imagename).convert('L')
    imgwidth, imgheight = im.size
    window_number=1
    radius=1
    num_points=8*radius
    lbp_hist_list=[]
    for i in range(0,imgheight,100):
        for j in range(0,imgwidth,100):
            box = (j, i, j+100, i+100)
            a = im.crop(box)
            final_img_path=os.path.join(image_store_path,imagename[:-4])
            if not os.path.exists(final_img_path):
                os.makedirs(final_img_path)
            a.save(final_img_path+"/"+str(window_number)+".jpg")
            window_number +=1
    filelist = [f for f in os.listdir(image_store_path+"\\"+imagename[:-4]) if f.endswith(".jpg")]
    for file in filelist:
        img = Image.open(os.path.join(image_store_path+"\\"+imagename[:-4] , file))
        lbp = local_binary_pattern(img,num_points,radius,method='uniform')
        (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, num_points + 3),range=(0, num_points + 2))
        lbp_hist_list.extend(hist)
    print(lbp_hist_list)
    #Delete the cropped images to make this task independent.
    shutil.rmtree(final_img_path)

# Perform task2 for color moments as an individual task

def LBP_task2(image_store_path):
    if (os.path.exists('LBPFeature.csv')):
        os.remove('LBPFeature.csv')
    for imagename in os.listdir(directory):
        LBP_image_windows(image_store_path, directory, imagename, 100, 100)
        LBP(image_store_path, imagename)

# Find out similar images according to Euclidean Distance and show it.

def LBP_Similar(imageid,k):

    # creating a dictionary to store the image name and the similarity value
    distance = {}
    with open(csv_path+'LBPFeature.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if (imageid == row[0]):
                cm = json.loads(row[1])
    with open(csv_path+'LBPFeature.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            dist = (sum([(a - b) ** 2 for a, b in zip(cm, json.loads(row[1]))]))**0.5
            distance.update({row[0]: dist})
    sorted_x = sorted(distance.items(), key=lambda x: x[1])

    # storing image name and similarity value in 2 lists
    img = [lis[0] for lis in sorted_x[:k]]
    dist = [lis[1] for lis in sorted_x[:k]]
    for i in range (0,k):
        im = Image.open(directory + img[i])
        im.show()
        print(img[i])
        print(dist[i])

# Perform task3 individually for LBP individually

def LBP_task3(image_store_path,imageid,k):
    if (os.path.exists('LBPFeature.csv')):
        os.remove('LBPFeature.csv')
    for imagename in os.listdir(directory):
        LBP_image_windows(image_store_path, directory, imagename, 100, 100)
        LBP(image_store_path, imagename)
    LBP_Similar(imageid, k)
def HOG_feat(imageid):
    img= cv2.imread(os.path.join(directory,imageid))
    width = int(img.shape[1]/10)  #downsampling the image from 10-1
    height = int(img.shape[0]/10)   #downsampling the height from 10-1
    resized_img = cv2.resize(img,(width,height), interpolation = cv2.INTER_AREA)  #store the resized image
    (h, _) = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2),transform_sqrt=False, block_norm="L2-Hys", visualize=True)  #Applying HOG funtion to obtain features of HOG
    h= list(h)  #converting the features into list
    return h
def HOG_1(imageid): #Task-1 of HOG Vector model
    h=HOG_feat(imageid)
    with open('feature_HOG.csv' , 'a') as csvfile:  #Storing the features
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([imageid, h])
        
def HOG_2(img_fold):  #Task-2 of HOG features
    filename = 'feature_HOG.csv'
    f = open(filename, "w+")
    f.close()
    for filename in os.listdir(img_fold):
        if filename.endswith(".jpg"):
            HOG_1(filename)
            
def HOG_3(imageid,k):   #Task-3 of HOG features
    HOG_2(directory)
    feat= HOG_feat(imageid)  #converting the features into list
    reader = csv.reader(csvfile, delimiter=',')
    f={}
    with open('feature_HOG.csv', 'r') as csvfile:
        for row in reader:
            if any(field.strip() for field in row):
                distance_img = (sum([(a - b) ** 2 for a, b in zip(feat, json.loads(row[1]))]))**0.5     #Euclidean Distance
                f.update({row[0]: distance_img})
    retrievedimg=sorted(f.items(), key = lambda x : x[1])       #sorting the Images
    sim_img=retrievedimg[:k]
    img = [lis[0] for lis in retrievedimg[:k]]
    dist = [lis[1] for lis in retrievedimg[:k]]
    for i in range (0,k):
        print(img[i])
        print(dist[i])
    for i in range(0,k):
        p=sim_img[i][0]
        image = Image.open(directory+p) 
        image.show()   
    
def SIFT_1(imageid):
    img = cv2.imread(directory+imageid)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descriptors) = sift.detectAndCompute(gray, None)
    return descriptors

def SIFT_2():
    for imageid in os.listdir(directory):
        descriptors=SIFT_1(imageid)
        if not os.path.exists(csv_path +'sift_csv'):
            os.mkdir(csv_path+'sift_csv')
        csv_file = csv_path+'sift_csv\\'+imageid[:-3] + 'csv'
        with open(csv_file, 'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerows(descriptors)
        f.close()
def SIFT_3(imageid,k):
    test_descriptors=SIFT_1(imageid)
    distance_matrix = sift_test(test_descriptors)
    for i in range(k):
        print("The image is ", distance_matrix[i][0], "Distance is ", distance_matrix[i][1])
        im = Image.open(directory + "/" + distance_matrix[i][0])
        im.show()
    print("SIFT Done")

def sift_test(test_sift_features):
    sift_matrix = []
    for csvfile in os.listdir(csv_path+"sift_csv"):
        #print(csvfile)
        with open(csv_path+"sift_csv\\"+csvfile, 'r') as f:
            sift_list = list(csv.reader(f))
        distance = 0.0
        for i in range(len(test_sift_features)):
            dist = []
            for j in range(len(sift_list)):
                dist.append(euclidean_distance(test_sift_features[i], sift_list[j]))
            distance += min(dist)
        sift_matrix.append([csvfile[:-3] + 'jpg', distance])
    sift_matrix = sorted(sift_matrix, key=lambda x: x[1])
    return sift_matrix

#Distance Measure Used Euclidean
def euclidean_distance(single_feature_vector,test_lbp_features):
    return math.sqrt(sum((float(a)-float(b))**2 for a,b in zip(single_feature_vector,test_lbp_features)))

    
# Performing all the tasks for both the models
if __name__ == '__main__':
    model = input("Provide the model you want to work with lbp/cm/HOG/SIFT: ")
    task = input("Tell us which task you want to perform: ")
    global CM_image_store_path
    #CM_image_store_path = input("Enter the folder: ")
    if (model.lower() == 'lbp'):
        if (task.lower()) =='task1':
            imageid = input("Provide the image name: ")
            LBP_task1(CM_image_store_path,imageid)
        elif (task.lower()) == 'task2':
            #img_store_path=("Provide the folder name : ")
            LBP_task2(CM_image_store_path)
        elif (task.lower()) == 'task3':
            imageid = input("Provide the image name: ")
            k = int(input("number of similar images you want to show: "))
            LBP_task3(CM_image_store_path,imageid,k)
        else:
            print("Wrong Choice!!!")
    elif (model.lower() == 'cm'):
        CM_image_store_path = input("Enter the folder: ")
        if (task.lower()) == 'task1':
            imageid = input("Provide the image name: ")
            CM_task1(CM_image_store_path, imageid)
        elif (task.lower()) == 'task2':
            CM_task2(CM_image_store_path)
        elif (task.lower()) == 'task3':
            imageid = input("Provide the image name: ")
            k = int(input("number of similar images you want to show: "))
            CM_task3(CM_image_store_path,imageid, k)
        else:
            print("Wrong Choice!!!")
    elif (model.lower() == 'hog'):
        if(task.lower()=='task1'):
            imageid=input("enter the image name\n")
            HOG_1(imageid)
        elif(task.lower()=='task2'):
            HOG_2(directory)
        elif(task.lower()=='task3'):
            imageid=input("Provide the image name:")    #input the image_id to which similarity has to be checked
            k=int(input("number of similar images you want to show:"))            
            HOG_3(imageid, k)                          
        else:
            print("Wrong Choice!!!")

    elif (model.lower() == 'sift'):
        if(task.lower()=='task1'):
            imageid=input("enter the image name\n")
            SIFT_1(imageid)
        elif(task.lower()=='task2'):
            SIFT_2()
        elif(task.lower()=='task3'):
            imageid=input("Provide the image name:")    #input the image_id to which similarity has to be checked
            k=int(input("number of similar images you want to show:"))
            SIFT_3(imageid, k)
        else:
            print("Wrong Choice!!!")
    else:
        print("Wrong Choice!!!")
#C:\\Users\\Lenovo\Desktop\\Downloads\\CSE 515 Fall19 - Smaller Dataset\\LBP(CM)
#Hand_0011684.jpg