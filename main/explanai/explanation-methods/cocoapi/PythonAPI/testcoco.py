from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

dataDir='main/cocoapi'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType) # annotations

# initialize COCO api for instance annotations
coco=COCO(annFile)

catIds = coco.getCatIds(catNms=['bus']) # on recup l'id de la categorie bus
imgIds = coco.getImgIds(catIds=catIds) # on recup l'id des images qui contiennent des bus
imgIds = imgIds[0] # on recup l'id de la premiere image qui contient un bus
img = coco.loadImgs(imgIds)[0] # on recup l'image, index 0 car on a qu'une image dans une liste

# use url to load image
I = io.imread(img['coco_url']) # easier to load image from url
# save the image in the current folder
#io.imsave('bus.jpg', I)

# display image without area
"""plt.axis('off')
plt.imshow(I)
plt.show()"""

# load and display instance annotations
#plt.imshow(I)
plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None) # on recup l'id de l'annotation de l'image
anns = coco.loadAnns(annIds) # on recup l'annotation de l'image
print(len(anns)) # on affiche le nombre d'annotation de l'image
temp_list = [] # temp liste because showanns waits a list and i want to show only one annotation
temp_list.append(anns[0]) # we add the first annotation to the list
print(temp_list) # on affiche l'annotation 1 de l'image
#coco.showAnns(anns) # on affiche l'annotation de l'image
#plt.show()

# test
# create a black image with the size of image I
img = np.zeros((I.shape[0], I.shape[1], 3), np.uint8)
#plt.imshow(img)
#coco.showAnns(temp_list) # on affiche l'annotation 1 de l'image
# extract the mask of the annotation
mask = coco.annToMask(temp_list[0]) # numpy 2D array of the mask
print(mask.shape) # on affiche la taille du masque pour vérifier la cohérence, =307200 (640*480) ce qui correspond bien au nb de 1 et 0

# count the number of ones and zeros in the mask
ones = np.count_nonzero(mask == 1) 
zeros = np.count_nonzero(mask == 0)
print("Number of ones: ", ones) # nb de pixels du masque
print("Number of zeros: ", zeros) # nb de pixels hors masque

plt.imshow(mask)
plt.show()