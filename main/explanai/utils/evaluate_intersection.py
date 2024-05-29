from cocoapi.PythonAPI.pycocotools.coco import COCO
import skimage.io as io
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

#to comment

def evaluate(id, catNms, mask):
    print(os.getcwd())
    dataType='val2017'
    dataDir='main/explanai/cocoapi'
    dataType='val2017'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType) # annotations

    coco=COCO(annFile)

    catIds = coco.getCatIds(catNms=catNms) # on recup l'id des categories passées en paramètre (une ou plusieurs catégories)

    annIds = coco.getAnnIds(imgIds=id, catIds=catIds, iscrowd=None) # on recup l'id de l'annotation de l'image
    anns = coco.loadAnns(annIds) # on recup l'annotation de l'image
    #print(anns) # interessant, on a les categories id pour savoir si c'est personne ou bus

    result_dict = {}

    for ann in anns:
        coco_mask = coco.annToMask(ann) # numpy 2D array of the mask

        # iterate over the method mask and the coco mask to get the intersection
        intersection = 0
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if np.all(coco_mask[i][j]) == 1 and np.all(mask[i][j]) != 0:
                    intersection += 1
                
        coco_pixels = np.ceil(ann['area'])

        # ratio compared to coco mask
        percentage = (intersection / coco_pixels)*100
        
        result_dict[str(ann['category_id'])+'-'+str(ann['id'])] = percentage

        """plt.imshow(coco_mask)
        plt.show()
        plt.imshow(mask)
        plt.show()"""
    
    return result_dict

#evaluate(329447, ['dog'], None) #test