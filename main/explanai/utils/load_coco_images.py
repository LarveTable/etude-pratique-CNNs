from cocoapi.PythonAPI.pycocotools.coco import COCO
import skimage.io as io
from tqdm import tqdm
import os
import random

#to comment

def download(dataType='val2017', catNms=['dog'], number_of_images=5, randomized=True):
    dataDir='main/explanai/cocoapi' 
    dataType='val2017'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType) # annotations

    # initialize COCO api for instance annotations
    coco=COCO(annFile)

    catIds = coco.getCatIds(catNms=catNms) # on recup l'id des categories passées en paramètre (une ou plusieurs catégories)
    imgIds = coco.getImgIds(catIds=catIds) # on recup l'id des images qui contiennent les catégories passées en paramètre

    directories_check(['main/data/images'])

    if len(imgIds) == 0:
        print("No images found with the categories: ", catNms)
        return None
    else:
        print("\nDownloading images...")
        if not randomized:
            for i in tqdm(range(number_of_images)):
                img = coco.loadImgs(imgIds[i])[0]
                I = io.imread(img['coco_url'])
                io.imsave('main/data/images/'+str(img['id'])+'-'+str(catIds)+'.jpg', I)
        else:
            for i in tqdm(range(number_of_images)):
                i = random.randint(0, len(imgIds)-1)
                img = coco.loadImgs(imgIds[i])[0]
                I = io.imread(img['coco_url'])
                io.imsave('main/data/images/'+str(img['id'])+'-'+str(catIds)+'.jpg', I)

def directories_check(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

#download() #test