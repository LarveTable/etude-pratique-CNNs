from skimage import io
from PIL import Image
from torchvision import transforms
from torchvision import datasets
from torch.utils import data
import os

#to comment

def process_dataset(path):

    # image folder path
    folder_path = path+'/images'

    # get the list of all images in the folder, excluding those starting with "."
    files = [file_name for file_name in os.listdir(folder_path) if not file_name.startswith(".")]

    if (files != []):
        
        # sort the files alphabetically
        files = sorted(files)

        # the transformation to apply to every image in the dataset
        transform = transforms.Compose([transforms.Resize((224, 224)), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # define the images dataset
        dataset = datasets.ImageFolder(root=path, transform=transform)

        # define the dataloader to load the images
        dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)

        # iterator for enumeration
        iterateur = iter(dataloader)

        return iterateur, files

    else:
        print("No images to process, please add them in the data/images folder.")
        return None, None
    
def process_one(img):

    # the transformation to apply to every image in the dataset
    transform = transforms.Compose([transforms.Resize((224, 224)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    image = Image.fromarray(img)
    image = transform(image)

    image = image.unsqueeze(0)

    return image
