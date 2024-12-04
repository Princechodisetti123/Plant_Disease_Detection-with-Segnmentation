import warnings
warnings.filterwarnings('ignore')
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import random
import math
import re
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from mrcnn import utils
from mrcnn import visualize
from mrcnn .visualize import display_differences
from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances
#from mrcnn import model as modellib
from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
from PIL import ImageTk
#import custom

# Root directory of the project
ROOT_DIR = "D:/real_world_projects/Plant-Disease-Detection-Using-Mask-R-CNN-main/Plant-Disease-Detection-Using-Mask-R-CNN-main"

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")


WEIGHTS_PATH = "D:\\real_world_projects\\Plant-Disease-Detection-Using-Mask-R-CNN-main\\Plant-Disease-Detection-Using-Mask-R-CNN-main\\mask_rcnn_object_0033.h5"   # change it
class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    IMAGES_PER_GPU = 1

    # 28 classes  # Background + Apple_Black_Rot, Apple_Healthy, Apple_Rust, Apple_Scab, Blueberry_Healthy, Corn_Common_Rust,
    #Corn_Gray_Leaf_Spot,Corn_Healthy, Grape_Black_Measles, Grape_Black_Rot, Grape_Healthy, Peach_Bacterial_Spot, Peach_Healthy,
    #Pepper_Bell_Bacterial_Spot, Pepper_Bell_Healthy, Pepper_Early_Blight, Potato_Healthy, Potato_Late_Blight, Rasberry_Healthy,
    #Soybean_Healthy,Strawberry_Leaf_Scroch,Strawberry_Healthy,Tomato_Bacterial_Spot,Tomato_Early_Blight, Tomato_Healthy,
    #Tomato_Late_Blight,Tomato_Leaf_Spot,Tomato_Target_Spot
    
    NUM_CLASSES = 1 + 17

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
config = CustomConfig()
CUSTOM_DIR = os.path.join(ROOT_DIR, "/dataset/")
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 #GPU with 24GB memory, which can fit 4 images. Adjust down if you use a smaller GPU.
    DETECTION_MIN_CONFIDENCE = 0.7

config = InferenceConfig()
config.display()
# Code for Customdataset class. Same code is present in custom.py file also



# Code for Customdataset class. Same code is present in custom.py file also

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
      
      
        # Add classes. We have only one class to add.
        self.add_class("object", 1, "Apple_Black_Rot")
        self.add_class("object", 2, "Apple_Healthy")
        self.add_class("object", 3, "Apple_Rust")
        self.add_class("object", 4, "Apple_Scab")
        self.add_class("object", 5, "Blueberry_Healthy")
        #self.add_class("object", 6, "Corn_Common_Rust")
        #self.add_class("object", 7, "Corn_Gray_Leaf_Spot")
        #self.add_class("object", 8, "Corn_Healthy")
        self.add_class("object", 6, "Grape_Black_Measles")
        self.add_class("object", 7, "Grape_Black_Rot")
        self.add_class("object", 8, "Grape_Healthy")
        self.add_class("object", 9, "Peach_Bacterial_Spot")
        self.add_class("object", 10, "Peach_Healthy")
        self.add_class("object", 11, "Pepper_Bell_Bacterial_Spot")
        #self.add_class("object", 15, "Pepper_Bell_Healthy")
        #self.add_class("object", 12, "Pepper_Early_Blight")  #potato_Early_Blight
        self.add_class("object", 12, "Potato_Healthy")
        self.add_class("object", 13, "Potato_Late_Blight")
        self.add_class("object", 14, "Raspberry_Healthy")
        self.add_class("object", 15, "Soybean_Healthy")
        #self.add_class("object", 21, "Strawberry_Leaf_Scroch")
        #self.add_class("object", 22, "Strawberry_Healthy")
        #self.add_class("object", 23, "Tomato_Bacterial_Spot")
        #self.add_class("object", 24, "Tomato_Early_Blight")   
        self.add_class("object", 16, "Tomato_Healthy")
        self.add_class("object", 17, "Tomato_Late_Blight")

     
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        # We mostly care about the x and y coordinates of each region
        
        #annotations1 = json.load(open('D:/python3.6.8_tensorflow_1.14_env/maskrcnn_leave_disease_detection/dataset/train/train.json'))
        annotations1 = json.load(open(os.path.join(dataset_dir, 'train.json')))
        #keep the name of the json files in the both train and val folders
        
        # print(annotations1)
        annotations = list(annotations1.values())  # don't need the dict keys



        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes']['Names'] for s in a['regions']]
            print("objects:",objects)
            name_dict = {"Apple_Black_Rot": 1,"Apple_Healthy": 2,"Apple_Rust": 3,"Apple_Scab":4,"Blueberry_Healthy":5,"Grape_Black_Measles":6,"Grape_Black_Rot":7,"Grape_Healthy":8,"Peach_Bacterial_Spot":9,"Peach_Healthy":10,"Pepper_Bell_Bacterial_Spot":11,"Potato_Healthy":12,"Potato_Late_Blight":13,"Raspberry_Healthy":14,"Soybean_Healthy":15,"Tomato_Healthy":16,"Tomato_Late_Blight":17}
            #name_dict = {"Apple_Black_Rot": 1,"Apple_Healthy": 2,"Apple_Rust": 3,"Apple_Scab":4,"Blueberry_Healthy":5,"Corn_Common_Rust":6,"Corn_Gray_Leaf_Spot":7,"Corn_Healthy":8,"Grape_Black_Measles":9,"Grape_Black_Rot":10,"Grape_Healthy":11,"Peach_Bacterial_Spot":12,"Peach_Healthy":13,"Pepper_Bell_Bacterial_Spot":14,"Pepper_Bell_Healthy":15,"Pepper_Early_Blight":16,"Potato_Healthy":17,"Potato_Late_Blight":18,"Rasberry_Healthy":19,"Soybean_Healthy":20,"Strawberry_Leaf_Scroch":21,"Strawberry_Healthy":22,"Tomato_Bacterial_Spot":23,"Tomato_Early_Blight":24,"Tomato_Healthy":25,"Tomato_Late_Blight":26,"Tomato_Leaf_Spot":27,"Tomato_Target_Spot":28}
            #name_dict = {"Horse": 1,"Man": 2} #,"xyz": 3}
            # key = tuple(name_dict)
            num_ids = [name_dict[b] for b in objects]
     
            # num_ids = [int(n['Event']) for n in objects]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            print("numids",num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
                )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Dog-Cat dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids #np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)




# Inspect the model in training or inference modes values: 'inference' or 'training'
TEST_MODE = "inference"
ROOT_DIR = "D:/real_world_projects/Plant-Disease-Detection-Using-Mask-R-CNN-main/Plant-Disease-Detection-Using-Mask-R-CNN-main"

def get_ax(rows=1, cols=1, size=6):
  """Return a Matplotlib Axes array to be used in all visualizations in the notebook.  Provide a central point to control graph sizes. Adjust the size attribute to control how big to render images"""
  _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
  return ax
# Load validation dataset
# Must call before using the dataset
CUSTOM_DIR = "D:/real_world_projects/Plant-Disease-Detection-Using-Mask-R-CNN-main/Plant-Disease-Detection-Using-Mask-R-CNN-main/dataset"
dataset = CustomDataset()
dataset.load_custom(CUSTOM_DIR, "val")
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
config = CustomConfig()
#LOAD MODEL. Create model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load COCO weights Or, load the last model you trained
weights_path = WEIGHTS_PATH
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)




#Now, we are ready for testing our model on any image.

#RUN DETECTION
'''image_id = random.choice(dataset.image_ids)
#image_id = 'D:/MaskRCNN-aar/Dataset/val/1.jfif'
print("image id is :",image_id)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,dataset.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)
print("result  :",results)

# Display results
# ax = get_ax(1)
# r1 = results1[0]
# visualize.display_instances(image1, r1['rois'], r1['masks'], r1['class_ids'],
# dataset.class_names, r1['scores'], ax=ax, title="Predictions1")


# Display results
x = get_ax(1)
r = results[0]
ax = plt.gca()
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],dataset.class_names, r['scores'],title="Predictions",ax=ax)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)'''



'''path_to_new_image = 'D:/real_world_projects/Plant-Disease-Detection-Using-Mask-R-CNN-main/Plant-Disease-Detection-Using-Mask-R-CNN-main/dataset2/PlantVillage/val/Apple___healthy/0c55b379-c6e7-4b89-959f-abc506fed437___RS_HL 5927.JPG'
image1 = mpimg.imread(path_to_new_image)

# Run object detection
print("len:",len([image1]))
results1 = model.detect([image1], verbose=1)
print("result",results1[0])
# Display results
ax = get_ax(1)
r1 = results1[0]
print(results1)
visualize.display_instances(image1, r1['rois'], r1['masks'], r1['class_ids'],dataset.class_names, r1['scores'],title="Predictions",ax=ax)
'''

'''import requests

def send_sms(number, message) :
    url = 'https://www.fast2sms.com/dev/bulkV2'
    params ={
    'authorization':'9FloAiVR2jhY5MUza3XpQwkWbHDI7ve8EGLdZ1Ng4rBuJxn6syZ0g2IpB6RMxev58YVPmUH7clskKj4W',
    'sender_id':'FSTSMS',
    'message':message,
    'language':'english',
    'route':'p',
    'numbers':number
    }
    response = requests.get(url, params=params)
    dic = response.json()#json will give dict
    print(dic)
    #return dic.get('return')'''



import tkinter
from tkinter import *
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tkinter import *
from PIL import ImageTk
from tkinter import messagebox
import mysql.connector as sql


def output_main():
    #global number2
    main = tkinter.Tk()
    main.title("Plant Diesases Identification And Tracking")
    main.geometry("1300x650")
    bgimage=ImageTk.PhotoImage(file='D:\\real_world_projects\\Plant-Disease-Detection-Using-Mask-R-CNN-main\\Plant-Disease-Detection-Using-Mask-R-CNN-main\\final-img.jpg')

    def change_cursor_to_hand(event):
        canvas1.config(cursor="hand2")
    def change_cursor_to_arrow(event):
        canvas1.config(cursor="")

    canvas1 = Canvas(main, width=1300, height=650)
    canvas1.create_image(0, 0, image=bgimage, anchor=NW)
    canvas1.create_text(620,40, text="PLANT DISEASES IDENTIFICATION AND TRACKING", fill="white", font=('times', 20, 'bold'))
    canvas1.pack(fill="both", expand=True)

    global file_path

    def upload_img():
        global file_path
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        img=mpimg.imread(file_path)
        input_fig = plt.figure(figsize=(6, 4))
        intput_ax = input_fig.add_subplot(111)
        intput_ax.imshow(img)  # Assuming image1 is the output image
        intput_ax.axis('off')
        intput_ax.set_title("Input Image")
        
        canvas = FigureCanvasTkAgg(input_fig, master=main)
        canvas.draw()
        canvas.get_tk_widget().place(x=20, y=150) 
        

    def predict():
        path_to_new_image = file_path
        image1 = mpimg.imread(path_to_new_image)

        # Run object detection
        print("len:",len([image1]))
        results1 = model.detect([image1], verbose=1)
        print("result",results1[0])
        # Display results
        ax = get_ax(1)
        r1 = results1[0]
        visualize.display_instances(image1, r1['rois'], r1['masks'], r1['class_ids'],dataset.class_names, r1['scores'],ax=ax)
        img1=mpimg.imread("D:\\real_world_projects\\Plant-Disease-Detection-Using-Mask-R-CNN-main\\Plant-Disease-Detection-Using-Mask-R-CNN-main\\output.png")
        output_fig = plt.figure(figsize=(6, 4))
        output_ax = output_fig.add_subplot(111)
        output_ax.imshow(img1)  # Assuming image1 is the output image
        output_ax.axis('off')
        output_ax.set_title("Output Image")
        canvas = FigureCanvasTkAgg(output_fig, master=main)
        canvas.draw()
        canvas.get_tk_widget().place(x=650, y=150)
        #send_sms(number2,"disease")

    def close():
        main.destroy()


    uploadButton = Button(main, text="Upload Image",font=('times', 13, 'bold'),bg='PaleGreen4',activebackground='honeydew4', cursor='hand2',command=upload_img)
    uploadButton.place(x=200,y=100)


    processButton = Button(main, text="Predict Image",font=('times', 13, 'bold'),bg='PaleGreen4', activebackground='honeydew4',cursor='hand2',command=predict)
    processButton.place(x=900,y=100) 

    exitButton = Button(main, text="Exit", font=('times', 13, 'bold'),bg='PaleGreen4',activebackground='honeydew4',cursor='hand2',command=close)
    exitButton.place(x=620,y=580)

    main.mainloop()





def forgot_pass():
    def change_pass():
        if userentry.get()=='' or pass1.get()=='':
            messagebox.showerror('Error','All Fields are Required',parent=forgotwindow)
        elif pass1.get() != pass2.get():
            messagebox.showerror('Error','Password is Mismatched',parent=forgotwindow)
        else:
            con=sql.connect(host='localhost',user='root',password='Derick!@1221',database='farmerdb')
            mycursor=con.cursor()
            query='select * from farmerdata where farmername=%s'
            mycursor.execute(query,[(userentry.get())])
            row=mycursor.fetchone()
            if row ==None:
                messagebox.showerror('Error','Incorrect Username',parent=forgotwindow)
            else:
                query='update farmerdata set password=%s where farmername=%s'
                mycursor.execute(query,(pass1.get(),userentry.get()))
                con.commit()
                con.close()
                messagebox.showinfo('Success','Password is reset,Please login with new password',parent=forgotwindow)
                forgotwindow.destroy()

            


    forgotwindow = Toplevel()
    forgotwindow.title("change password")

    bgimage = ImageTk.PhotoImage(file='forgot.png')
    image=Label(forgotwindow,image=bgimage)
    image.grid()

    heading=Label(forgotwindow,text="Forgot Password",font=("Microsoft yanei UI Light", "15"),fg='green',bg='white',bd=0)
    heading.place(x=450,y=70)

    user=Label(forgotwindow,text="User Name:",font=("Microsoft yanei UI Light", "12"),bg='white',fg='darkgreen',bd=0)
    user.place(x=410,y=120)
    passlab=Label(forgotwindow,text="New Password:",font=("Microsoft yanei UI Light", "12"),bg='white',fg='darkgreen',bd=0)
    passlab.place(x=410,y=180)
    passlab1=Label(forgotwindow,text="Confirm Password:",font=("Microsoft yanei UI Light", "12"),bg='white',fg='darkgreen',bd=0)
    passlab1.place(x=410,y=240)

    userentry=Entry(forgotwindow,width=26,font=("Microsoft yanei UI Light",11),bg='DarkSeaGreen3',bd=0)
    userentry.place(x=430,y=150)


    pass1=Entry(forgotwindow,width=26,font=("Microsoft yanei UI Light",11),bg='DarkSeaGreen3',bd=0)
    pass1.place(x=430,y=210)

    pass2=Entry(forgotwindow,width=26,font=("Microsoft yanei UI Light",11),bg='DarkSeaGreen3',bd=0)
    pass2.place(x=430,y=270)


    submitbutton = Button(forgotwindow, width=20, height=2, text='SUBMIT', bd=0, bg='PaleGreen4', cursor='hand2', font=("Microsoft yanei UI Light", 11), fg='black', activebackground='#c7f0bd',command=change_pass)
    submitbutton.place(x=438, y=310)

    forgotwindow.mainloop()        
   
    
def cleardata():
    userentry.delete(0,END)
    passentry.delete(0,END)

def login_user():
    global number2
    if userentry.get()=='' or passentry.get()=='':
        messagebox.showerror('Error','All Fields Required')
    else:
        try:
            con=sql.connect(host='localhost',user='root',password='Derick!@1221')
            mycursor=con.cursor()
        except:
            messagebox.showerror('Error','Connection is not Estalblished try Again')
            return
        query='use farmerdb'
        mycursor.execute(query)
        query='select * from farmerdata where farmername=%s and password=%s'
        mycursor.execute(query,(userentry.get(),passentry.get()))
        row=mycursor.fetchone()
        if row==None:
            messagebox.showerror('Error','Invalid Username or Password')
        else:
            messagebox.showinfo('Success','welcome')
            #query='select * from farmerdata where farmername=%s'
            #mycursor.execute(query,[(userentry.get())])
            #row=mycursor.fetchone()
            #number2=row[3]
            cleardata()
            loginwindow.destroy()
            output_main()


def signup():
    loginwindow.destroy()
    import signup1

def user_enter(event):
    if userentry.get()=='Username':
        userentry.delete(0,END)

def pass_enter(event):
    if passentry.get()=='Password':
        passentry.delete(0,END)


loginwindow=Tk()
loginwindow.geometry("874x494+50+50")
loginwindow.title("Login page")

bgimage=ImageTk.PhotoImage(file='D:\\real_world_projects\\Plant-Disease-Detection-Using-Mask-R-CNN-main\\Plant-Disease-Detection-Using-Mask-R-CNN-main\\bg1.jpg')
bgLabel=Label(loginwindow,image=bgimage)
bgLabel.place(x=0,y=0)

heading=Label(loginwindow,text="FARMER LOGIN",font=("Microsoft yanei UI Light", "19"),fg='green',bg='white',bd=0)
heading.place(x=572,y=70)

userentry=Entry(loginwindow,width=26,font=("Microsoft yanei UI Light",11),fg='green',bd=0)
userentry.place(x=540,y=120)
userentry.insert(0,'Username')
userentry.bind('<FocusIn>',user_enter)
Frame(loginwindow,width=250,height=2,bg='green').place(x=540,y=142)

passentry=Entry(loginwindow,width=26,font=("Microsoft yanei UI Light",11),fg='green',bd=0)
passentry.place(x=540,y=180)
passentry.insert(0,'Password')
passentry.bind('<FocusIn>',pass_enter)
Frame(loginwindow,width=250,height=2,bg='green').place(x=540,y=200)

forgetbutton=Button(loginwindow,text='Forgot Password?',bd=0,bg='white',cursor='hand2',font=("Microsoft yanei UI Light",11),fg='green',activebackground='white',command=forgot_pass)
forgetbutton.place(x=700,y=222)

loginbutton=Button(loginwindow,width=20,height=2,text='LOGIN',bd=0,bg='green',cursor='hand2',font=("Microsoft yanei UI Light",11),fg='white',activebackground='white',command=login_user)
loginbutton.place(x=575,y=265)

orlabel=Label(loginwindow,text='-------------- OR --------------',font=('Open Sans',16),fg='green',bg='white')
orlabel.place(x=550,y=330)

signuplabel=Label(loginwindow,text="Don't have an Account?",font=('Open Sans',10,'bold'),fg='green',bg='white')
signuplabel.place(x=530,y=385)

newbutton=Button(loginwindow,text='Create new one',bd=0,bg='white',cursor='hand2',font=("Open Sans",11,'underline'),fg='firebrick1',activebackground='white',command=signup)
newbutton.place(x=690,y=385)

loginwindow.mainloop()
