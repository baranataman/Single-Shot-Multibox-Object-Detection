from pathlib import Path
import time
from tensorflow.keras import backend as K
import numpy as np
from models.keras_ssd300 import ssd_300
import cv2
from skimage.transform import resize as imresize

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

import matplotlib.pyplot as plt

# Set the network image size.
img_height = 300
img_width = 300


K.clear_session() # Clear previous models from memory.
# Define the model: Single Shot Multi Box Detector 300
model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], 
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=10,
                nms_max_output_size=400)

# Load the trained weights into the model.
weights_path = Path('VGG_VOC0712_SSD_300x300_iter_120000.h5')
model.load_weights(str(weights_path), by_name=True)


classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

# Create a camera instance
#cap =  cv2.VideoCapture(0)
cap = cv2.VideoCapture("sample_video.mp4") 
confidence_threshold = 0.0
delay = 1

while True:
    ret, img = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # restart video
        continue

    start = time.time()

    img_resized = imresize(img, (img_height, img_width), preserve_range ='True')
    
    image = np.array([img_resized])
    
    # Detect the objects on the frame and mask the undetected objects from the list of classes
    pred = model.predict(image)
    idx = pred[:,:,1] > 0
    pred = pred[idx]
    
    # for each detected object, mark them on the frame one by one
    for obj in pred:
        ratio_y = (np.size(img,0)/img_height)
        ratio_x = (np.size(img,1)/img_width)
        
        xmin = int(obj[2] * ratio_x)
        ymin = int(obj[3] * ratio_y)
        xmax = int(obj[4] * ratio_x)
        ymax = int(obj[5] * ratio_y)
        
        cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (255,0,0), 2)
        cv2.putText(img, classes[int(obj[0])], (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX,
                          fontScale=0.8, color=(0, 255, 255))
    

    cv2.imshow('Video feed', img)


        # q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
       
cv2.destroyAllWindows()

