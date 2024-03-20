import json
import numpy as np
import torch
from torchvision import models
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
import shap

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class shap_explainer:
    def __init__(self, model, model_layer, image_size=(224, 224)):
        self.model = model
        self.model_layer = model_layer
        self.image_size = image_size

    def normalize(self, image):
        if image.max() > 1:
            image /= 255
        image = (image - mean) / std
        return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()

    def explain_image(self, img_path, ranked_outputs=3, nsamples=5):
        img = image.load_img(img_path, target_size=self.image_size)
        im = image.img_to_array(img)
        im = np.expand_dims(im, axis=0)

        X = shap.datasets.imagenet50()[0]
        X[0] = im
        X /= 255
        to_explain = X[[0]]

        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        fname = shap.datasets.cache(url)
        with open(fname) as f:
            class_names = json.load(f)

        e = shap.GradientExplainer((self.model, self.model_layer), self.normalize(X))
        shap_values, indexes = e.shap_values(
            self.normalize(to_explain), ranked_outputs=ranked_outputs, nsamples=nsamples
        )

        index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)
        shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

        shap.image_plot(shap_values, to_explain, index_names)

# Example usage:
model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
model.eval()
explainer = shap_explainer(model, model.features[7])
explainer.explain_image("../data/Trousse.jpg")
