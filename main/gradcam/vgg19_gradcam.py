import cv2
import torch
import torch.nn as nn
from torchvision.models import vgg19
from torchvision.models import VGG19_Weights
import numpy as np
import time
import os

def gradcam_process(image_to_process, file_name, neural_network) :

    # dissect the vgg19 network

    class VGG(nn.Module):
        def __init__(self):
            super(VGG, self).__init__()
        
            # get the pretrained VGG19 network
            self.vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        
            # disect the network to access its last convolutional layer
            self.features_conv = self.vgg.features[:36]
        
            # get the max pool of the features stem
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
            # get the classifier of the vgg19
            self.classifier = self.vgg.classifier
        
            # placeholder for the gradients
            self.gradients = None
    
        # hook for the gradients of the activations
        def activations_hook(self, grad):
            self.gradients = grad
        
        def forward(self, x):
            x = self.features_conv(x)
        
            # register the hook
            h = x.register_hook(self.activations_hook)
        
            # apply the remaining pooling
            x = self.max_pool(x)
            x = x.view((1, -1))
            x = self.classifier(x)
            return x
    
        # method for the gradient extraction
        def get_activations_gradient(self):
            return self.gradients
    
        # method for the activation exctraction
        def get_activations(self, x):
            return self.features_conv(x)

    # initialize the VGG model
    vgg = VGG()

    # set the evaluation mode
    vgg.eval()

    # get the most likely prediction of the model
    pred = vgg(image_to_process)

    # get the top 5 class_ids whith their matched probability
    pred2 = pred.squeeze(0).softmax(0)
    top_5_conf, i = pred2.topk(5)

    pred_res = "Prediction for "+file_name+" : \n"

    # for each class_id, add them to the result with its associated score
    itr = 0
    for x in i:
        # ask the included categories to match the class_id with a label
        category_name = VGG19_Weights.DEFAULT.meta["categories"][x.item()]
        score = top_5_conf[itr].item()
        # write the prediction into the pred_res variable
        pred_res += f"- {category_name} / {100 * score:.1f}%\n"
        itr=itr+1

    start = time.time()

    # get the gradient of the output with respect to the parameters of the model
    pred[:, 386].backward()

    # pull the gradients out of the model
    gradients = vgg.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = vgg.get_activations(image_to_process).detach()

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
    
    # superimpose the heatmap on the image
    heatmap = np.asarray(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # pixelate the heatmap
    test = heatmap*0.4

    # count how many pixels have each color present in the image
    # unique_colors, counts = np.unique(test.reshape(-1, 3), axis=0, return_counts=True)
    # color_counts = dict(zip(map(tuple, unique_colors), counts))

    # get the max counts and associated color which corresponds to non affected pixels
    # max_color = max(color_counts, key=color_counts.get)
    # get the max count
    # max_count = color_counts[max_color]

    # count non affected pixels
    non_affected = 0
    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            if (test[i][j][0] == 51.2 and test[i][j][1] == 0 and test[i][j][2] == 0):
                non_affected += 1
                test[i][j][0] = 0
                test[i][j][1] = 0
                test[i][j][2] = 0

    # get the ratio of non affected pixels
    ratio = non_affected/(test.shape[0]*test.shape[1])
    print(f'Ratio of non affected pixels : {ratio:.3}')

    cv2.imshow('test', test)
    cv2.waitKey(0)

    superimposed_img = heatmap * 0.4 + img

    end = time.time()
    elapsed = end-start

    return superimposed_img, pred_res, elapsed