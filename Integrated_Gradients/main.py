import numpy as np
import torch
from torchvision import models
from torchvision.models import VGG19_Weights

import cv2
import torch.nn.functional as F
from utils import calculate_outputs_and_gradients, generate_entrie_images
from integrated_gradients import random_baseline_integrated_gradients
from visualization import visualize
import argparse
import os

parser = argparse.ArgumentParser(description='integrated-gradients')
parser.add_argument('--model-type', type=str, default='inception', help='the type of network')
parser.add_argument('--img', type=str, default='01.jpg', help='the images name')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Check and create directories for storing results
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/' + args.model_type):
        os.mkdir('results/' + args.model_type)
    
    # Start to create models based on the model type
    if args.model_type == 'inception':
        model = models.inception_v3(weights=models.Inception3_Weights.DEFAULT)
    elif args.model_type == 'resnet152':
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    elif args.model_type == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif args.model_type == 'vgg19':
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

    model.eval()
    

    

    # Read and preprocess the image
    img = cv2.imread('examples/' + args.img)
    if args.model_type == 'vgg19':
        img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = img[:, :, (2, 1, 0)]

    # Calculate the gradient and the label index
    
    gradients, label_index = calculate_outputs_and_gradients([img], model, None)
    category_name = VGG19_Weights.DEFAULT.meta["categories"][label_index]
    print("Prediction :" + category_name)
    gradients = np.transpose(gradients[0], (1, 2, 0))
    img_gradient_overlay = visualize(gradients, img, clip_above_percentile=99, clip_below_percentile=0, overlay=True, mask_mode=True)
    img_gradient = visualize(gradients, img, clip_above_percentile=99, clip_below_percentile=0, overlay=False)
  
    # Calculate the integrated gradients
    attributions = random_baseline_integrated_gradients(img, model, label_index, calculate_outputs_and_gradients,
                                                        steps=50, num_random_trials=10)
    img_integrated_gradient_overlay = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0,
                                                overlay=True, mask_mode=True)
    img_integrated_gradient = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0, overlay=False)
    
# Save each image separately instead of generating a composite image
outer_directory = "results"
inner_directory = category_name

# Full path of the inner directory
full_directory_path = os.path.join(outer_directory, args.model_type, inner_directory)

# Create the directories if they don't exist
if not os.path.exists(full_directory_path):
    os.makedirs(full_directory_path)

"""cv2.imwrite('results/' + args.model_type + '/' + inner_directory + '/original_' + args.img, img)
cv2.imwrite('results/' + args.model_type + '/' + inner_directory + '/gradient_overlay_' + args.img, np.uint8(img_gradient_overlay))
cv2.imwrite('results/' + args.model_type + '/' + inner_directory + '/gradient_' + args.img, np.uint8(img_gradient))
cv2.imwrite('results/' + args.model_type + '/' + inner_directory + '/integrated_gradient_' + args.img, np.uint8(img_integrated_gradient))
"""
cv2.imwrite('results/' + args.model_type + '/' + inner_directory + '/integrated_gradient_overlay_' + args.img, np.uint8(img_integrated_gradient_overlay))
