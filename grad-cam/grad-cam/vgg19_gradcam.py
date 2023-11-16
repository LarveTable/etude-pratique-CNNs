import cv2
import torch
import torch.nn as nn
from torchvision.models import vgg19
from torchvision.models import VGG19_Weights
import numpy as np

#String variableto store the predictions
pred_res = ""

def process_image(image_to_process, file_name) :

    global pred_res

    #dissect the vgg19 network

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

    #Get the class_id whith the most probability
    pred2 = pred.squeeze(0).softmax(0)
    class_id = pred2.argmax().item()
    score = pred2[class_id].item()

    #Get the matched category name from the matching class_id
    category_name = VGG19_Weights.DEFAULT.meta["categories"][class_id]

    #Write the prediction into the pred_res variable
    pred_res += ("Prediction for "+file_name+f" : {category_name} / {100 * score:.1f}%\n")

    #Facultatif
    print(f"Prediction : {category_name}: {100 * score:.1f}%")

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

    """
    #Only for demonstration
    # draw the heatmap
    plt.matshow(heatmap.squeeze())
    plt.show()
    """
    
    #interpolate
    img = cv2.imread('./data/images/'+file_name)
    
    """
    #Only for demonstration
    #show the image
    cv2.imshow('image', img)
    cv2.waitKey(0)
    """
    
    #Superimpose the heatmap on the image
    heatmap = np.asarray(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img

    #Save the resulted grad-cam image
    cv2.imwrite('./results/gradcam_'+file_name, superimposed_img) 

#Export all the results in a txt file
def export_preds_to_file():
    path = "./results/preds_results.txt"
    try:
        # Open the file in write mode
        with open(path, 'w') as file:
            # Write the string to the file
            file.write(pred_res)
        print(f'Successfully exported to {path}')
    except Exception as e:
        print(f'Error exporting to {path}: {e}')