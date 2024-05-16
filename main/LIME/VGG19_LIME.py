
import torch
import torch.nn as nn
from torchvision.models import vgg19
from torchvision.models import VGG19_Weights
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import os, json
from torchvision import models, transforms
from torch.autograd import Variable
# import torch.nn.functional as F
from torch.nn import functional as F
import cv2

from PIL import Image

import time

#to comment

from lime import lime_image
from skimage.segmentation import mark_boundaries

def lime_process(img, file_name, nn):

    class LimeVGG19:

        def __init__(self):
            self.model = models.vgg19(weights = VGG19_Weights.DEFAULT)
            self.idx2label, self.cls2label, self.cls2idx = self.load_class_labels()
            self.pill_transf = self.get_pil_transform()
            self.preprocess_transform = self.get_preprocess_transform()
            self.explainer = lime_image.LimeImageExplainer()


        def get_image(self, path):
            # with open(os.path.abspath(path), 'rb') as f:
            with open(path, 'rb') as f:
                with Image.open(f) as img:
                    return img.convert('RGB') 
                
        # img = get_image('./Images/Trousse.jpg')
        # plt.imshow(img)

        def get_input_transform(self):
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])       
            transf = transforms.Compose([
                # version originale : transforms.Resize((240, 240)), 
                transforms.Resize((224,224)),
                # pas besoin de centercrop si on resize directement : transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])    
            return transf

        def get_input_tensors(self, img):
            transf = self.get_input_transform()
            # unsqeeze converts single image to batch of 1
            return transf(img).unsqueeze(0)

    # model = vgg19(weights=models.VGG19_Weights.DEFAULT)

        def load_class_labels(self):
            idx2label, cls2label, cls2idx = [], {}, {}
            file_path = './main/LIME/imagenet_class_index.json'
            with open(file_path, 'r') as read_file:
                class_idx = json.load(read_file)
                idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
                cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
                cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}  
            return idx2label, cls2label, cls2idx 


        def batch_predict(self, images):
            self.model.eval()
            batch = torch.stack(tuple(self.preprocess_transform(i) for i in images), dim=0)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            batch = batch.to(device)
            
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1)
            return probs.detach().cpu().numpy()


        def get_pil_transform(self): 
            transf = transforms.Compose([
                transforms.Resize((224, 224))
            ])    
            return transf
        
        def get_preprocess_transform(self):
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]) # normalize concerne les valeurs des pixels    
                transf = transforms.Compose([
                    transforms.ToTensor(),
                    normalize
                ])    
                return transf  
        

        def explain_image(self, img_name):
            img_raw = self.get_image('main/data/images/'+img_name)

            # DEBUT TIMER : DUREE DE L'EXPLICATON 
            start_explanation_time = time.time()
            # explanation = self.explainer.explain_instance(np.array(self.pill_transf(img)), 
            # remplacer self.pillblabla par juste img_T
            explanation = self.explainer.explain_instance(np.array(self.pill_transf(img_raw)),
                                                        self.batch_predict,
                                                        top_labels=5,
                                                        hide_color=0,
                                                        num_samples=100)
            
            # FIN DU TIMER
            end_explanation_time = time.time()
            # CALCUL DE LA DUREE
            explanation_time = end_explanation_time - start_explanation_time
            #print("Temps pris pour l'explication : {:.2f} secondes".format(explanation_time))
            

            return explanation, explanation_time

        def visualize_explanation(self, explanation, img_name):
            
            start_timer = time.time()

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                        positive_only=True,
                                                        num_features=5,
                                                        hide_rest=False)
            img_boundry1 = mark_boundaries(temp / 255.0, mask)

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                        positive_only=False,
                                                        num_features=10,
                                                        hide_rest=False)
            img_boundry2 = mark_boundaries(temp / 255.0, mask)
            # FIN DU TIMER 
            end_timer = time.time()
            # CALCUL DU TEMPS TOTAL 
            visualization_time = end_timer - start_timer
            
            #print("Temps pris pour la visualisation : {:.2f} secondes".format(visualization_time))
            """plt.subplot(1, 2, 1)
            plt.imshow(img_boundry1)
            plt.subplot(1, 2, 2)
            plt.imshow(img_boundry1)
            plt.show()"""


            #To resize
            img_raw = cv2.imread('main/data/images/'+img_name)

            resized = cv2.resize(img_boundry2, (img_raw.shape[1], img_raw.shape[0]))
            resized_mask = cv2.resize(mask, (img_raw.shape[1], img_raw.shape[0]), interpolation=cv2.INTER_NEAREST)

            #Color
            colored = np.uint8(255 * resized)

            #Change color space (fucking thanks to GPT !!!)
            image_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

            filtered = np.zeros_like(img_raw)
            for i in range(resized_mask.shape[0]):
                for j in range(resized_mask.shape[1]):
                    if not np.all(resized_mask[i][j] == [0, 0, 0]):
                        filtered[i][j] = img_raw[i][j]

            return image_rgb, resized_mask, filtered

    # MÉTHODES PLUTOT LIÉES AU CLASSIFIEUR QU'À LIME DIRECTEMENT 
        def new_predict_fn(self, images):
                images = self.skimage_to_vgg(images)
                return self.predict_fn(images)

        def predict(self, img, file_name):
            self.model.eval()
            logits = self.model(img)
            probs = F.softmax(logits, dim=1)
            probs5 = probs.topk(5)
            pred_res = "Prediction for "+file_name+" : \n"
            for p, c in zip(probs5[0][0].detach().numpy(), probs5[1][0].detach().numpy()):
                pred_res += f"{self.idx2label[c]}: {p:.2%}\n"
            return pred_res

        def test_prediction(self, img):
            pill_transf = self.get_pil_transform()
            # preprocess_transform = self.get_preprocess_transform()
            test_pred = self.batch_predict([pill_transf(img)])
            return test_pred.squeeze().argmax()
        

    # PARTIE IMPORTANTE 
        # fonction qui appelle les autres fonctions essentielles :
        def lime_process_all(self, img, file_name, nn):
        
            image = lime_vgg.get_image(img)
            explanation = lime_vgg.explain_image(img)
            lime_vgg.visualize_explanation(explanation, file_name)


    # NEW MAIN !!!!!!!!!!!!!!!!!! : 
    #lime_vgg = LimeVGG19()
    #lime_vgg.lime_process_all('./Images/Trousse.jpg', 'test.txt', nn)



    # OLD MAIN :
    #lime_vgg = LimeVGG19()
    #img = lime_vgg.get_image('./Trousse.jpg')
    # Obtenir les prédictions top-5
    #predictions_top5 = lime_vgg.predict(img)
    # Obtenir la classe prédite
    # predicted_class = lime_vgg.test_prediction(img)
    # on demande d'expliquer la prédiciton de l'image dont le path est passé en paramètre : 
    #explanation = lime_vgg.explain_image('./Trousse.jpg')
    # on demande de visualiser l'explication de la prédiction via Lime 
    #lime_vgg.visualize_explanation(explanation, './Trousse.jpg')

    #lime process
    lime_vgg = LimeVGG19()
    # Obtenir les prédictions top-5
    predictions_top5 = lime_vgg.predict(img, file_name)
    # on demande d'expliquer la prédiciton de l'image dont le path est passé en paramètre : 
    explanation, explanation_time = lime_vgg.explain_image(file_name)
    # on demande de visualiser l'explication de la prédiction via Lime 
    output, mask, filtered_image = lime_vgg.visualize_explanation(explanation, file_name)

    return output, predictions_top5, explanation_time, mask, filtered_image
