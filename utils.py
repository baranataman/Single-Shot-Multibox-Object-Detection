#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
import matplotlib.pyplot as plt

def plotHistory(history):
    # Plot training/validation information
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(loc='upper right', prop={'size': 24})
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def predict(val_dataset, model, img_h, img_w):
    # Set the generator for the predictions.
    predict_generator = val_dataset.generate(batch_size=1,
                                             shuffle=False,
                                             transformations=[],
                                             label_encoder=None,
                                             returns={'processed_images',
                                                      'processed_labels',
                                                      'filenames'},
                                             keep_images_without_gt=False)
    
    
    # Generate samples
    batch_images, batch_labels, batch_filenames = next(predict_generator)
    
    i = 0  # Which batch item to look at
    
    print("Image:", batch_filenames[i])
    print()
    print("Ground truth boxes:\n")
    print(batch_labels[i])
    
    
    
    # Make a prediction
    y_pred = model.predict(batch_images)
    
    
    # Decode the raw prediction `y_pred`
    y_pred_decoded = decode_detections(y_pred,
                                       confidence_thresh=0.34,
                                       iou_threshold=0.5,
                                       top_k=200,
                                       normalize_coords=True,
                                       img_height=img_h,
                                       img_width=img_w)
    
    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_decoded[i])
    
    
    # Draw the predicted boxes onto the image
    plt.figure()
    plt.imshow(batch_images[i])
    plt.title('Green: ground truth, Blue: prediction')
    
    current_axis = plt.gca()
    
    #colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist() # Set the colors for the bounding boxes
    #classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light'] # Just so we can print class names onto the image instead of IDs
    
    # Draw the ground truth boxes in green (omit the label for more clarity)
    for box in batch_labels[i]:
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        #label = '{}'.format(classes[int(box[0])])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
        #current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})
    
    # Draw the predicted boxes in blue
    for box in y_pred_decoded[i]:
        xmin = box[-4]
        ymin = box[-3]
        xmax = box[-2]
        ymax = box[-1]
        #color = colors[int(box[0])]
        #label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='blue', fill=False, linewidth=2))  
        #current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
        
    plt.show()