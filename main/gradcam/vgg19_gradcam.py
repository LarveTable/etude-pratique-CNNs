import cv2
import torch
import torch.nn as nn
from torchvision.models import vgg19
from torchvision.models import VGG19_Weights
import numpy as np
import time
import os
#temp
import matplotlib.pyplot as plt

#to comment

def gradcam_process(image_to_process, file_name, neural_network, pred) :

    start = time.time()

    # maybe loop gradcam part for time elapsed

    # get the gradient of the output with respect to the parameters of the model
    pred[:, 386].backward()

    # pull the gradients out of the model
    gradients = neural_network.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = neural_network.get_activations(image_to_process).detach()

    # weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    
    # interpolate
    img = cv2.imread('main/data/images/'+file_name)
    
    # compute the heatmap
    heatmap = np.asarray(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_bw = heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    end = time.time()
    elapsed = end-start

    # pixelate the heatmap
    mask = heatmap*0.4 #explanations here : https://medium.com/@Coursesteach/computer-vision-part-13-multiply-by-a-scaler-60627d66c820
    mask_bw = heatmap_bw

    # count how many pixels have each color present in the image
    # unique_colors, counts = np.unique(test.reshape(-1, 3), axis=0, return_counts=True)
    # color_counts = dict(zip(map(tuple, unique_colors), counts))

    # get the max counts and associated color which corresponds to non affected pixels
    # max_color = max(color_counts, key=color_counts.get)
    # get the max count
    # max_count = color_counts[max_color]

    # count non affected pixels
    """non_affected = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (mask[i][j][0] == 51.2 and mask[i][j][1] == 0 and mask[i][j][2] == 0):
                non_affected += 1
                mask[i][j][0] = 0
                mask[i][j][1] = 0
                mask[i][j][2] = 0"""

    # get the ratio of non affected pixels
    # ratio = non_affected/(mask.shape[0]*mask.shape[1])
    #print(f'Ratio of non affected pixels : {ratio:.3}')

    #cv2.imshow('test', test)
    #cv2.waitKey(0)

    superimposed_img = heatmap * 0.4 + img
    
    final_mask = np.zeros_like(img)
    filtered_img = np.zeros_like(img)
    for i in range(mask_bw.shape[0]):
        for j in range(mask_bw.shape[1]):
            if not np.all(mask_bw[i][j] == [0, 0, 0]):
                filtered_img[i][j] = img[i][j]
                final_mask[i][j] = [255, 255, 255]

    #temp
    """plt.axis('off')
    plt.imshow(filtered_img)
    plt.show()"""

    return superimposed_img, elapsed, final_mask, filtered_img #change mask to mask_bw to get the black and white mask and don't forget to pixelate it