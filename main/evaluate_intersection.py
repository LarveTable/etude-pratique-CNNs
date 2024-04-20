from cocoapi.PythonAPI.pycocotools.coco import COCO
import skimage.io as io
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import cv2

def evaluate(id, catNms, gradcam_mask):
    dataType='val2017'
    dataDir='main/cocoapi'
    dataType='val2017'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType) # annotations

    coco=COCO(annFile)

    catIds = coco.getCatIds(catNms=catNms) # on recup l'id des categories passées en paramètre (une ou plusieurs catégories)

    annIds = coco.getAnnIds(imgIds=id, catIds=catIds, iscrowd=None) # on recup l'id de l'annotation de l'image
    anns = coco.loadAnns(annIds) # on recup l'annotation de l'image
    #print(anns) # interessant, on a les categories id pour savoir si c'est personne ou bus

    temp_list = [] # temp liste because showanns waits a list and i want to show only one annotation
    temp_list.append(anns[0]) # we add the first annotation to the list

    mask = coco.annToMask(temp_list[0]) # numpy 2D array of the mask

    # iterate over the gradcam mask and the coco mask to get the intersection

    plt.imshow(mask)
    plt.show()

evaluate(71226, ['dog', 'cat'], None)