from torchvision import transforms
from torchvision import datasets
from torch.utils import data
import os
from vgg19_gradcam import process_image
from rename_all_files import rename_files
from vgg19_gradcam import export_preds_to_file

# image folder path
print(os.getcwd())
folder_path = 'grad-cam/data/images'

# rename every file in numeric order to later match them with their prediction
rename_files(folder_path)

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
    dataset = datasets.ImageFolder(root='grad-cam/data/', transform=transform)

    # define the dataloader to load the images
    dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)

    # iterator for enumeration
    iterateur = iter(dataloader)

    # enumerate through the images and apply gradcam
    for file_name in files:
        print('Processing : '+file_name)
        img, _ = next(iterateur)
        process_image(img, file_name)

    # export results when done
    export_preds_to_file()

else:
    print("No images to process, please add them in the data/images folder.")