import torch
import torch.nn as nn
from torchvision.models import vgg19
from torchvision.models import VGG19_Weights

class VGGGradCAM(nn.Module):
    def __init__(self):
        super(VGGGradCAM, self).__init__()
    
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

class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase, self).__init__()
    
        # get the pretrained VGG19 network
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT)
    
        # use the full model as is
        self.model = self.vgg

    def forward(self, x):
        return self.model(x)
    
def init_model_vgg19(image_to_process, file_name, use_gradcam=False):
    if use_gradcam:
        # initialize the VGG model for Grad-CAM
        vgg = VGGGradCAM()
    else:
        # initialize the base VGG model
        vgg = VGGBase()

    # set the evaluation mode
    vgg.eval()

    # get the most likely prediction of the model
    pred = vgg(image_to_process)

    # get the first prediction class
    class_id = pred.argmax().item()

    # get the label
    first_class = VGG19_Weights.DEFAULT.meta["categories"][class_id]

    # get the top 5 class_ids with their matched probability
    pred2 = pred.squeeze(0).softmax(0)
    top_5_conf, i = pred2.topk(5)

    preds_top5 = "Prediction for "+file_name+" : \n"

    # for each class_id, add them to the result with its associated score
    itr = 0
    for x in i:
        # ask the included categories to match the class_id with a label
        category_name = VGG19_Weights.DEFAULT.meta["categories"][x.item()]
        score = top_5_conf[itr].item()
        # write the prediction into the preds_top5 variable
        preds_top5 += f"- {category_name} / {100 * score:.1f}%\n"
        itr=itr+1
    
    return vgg, preds_top5, first_class, pred
