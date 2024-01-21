# etude-pratique-CNNs
> Repository to share our codes and other useful files on our CNNs
> explanation project. By Adèle, Sulaymane, Taha, Yorick.

Repository for our practical study "A Comparative Study of Neural Network Explanation Methods" realized in 2023/2024. In this study, we have compared 4 explaning ai methods according to a group of metrics that we will define in the second part of this project.
Each folder contains an implementation of a method explaining a prediction of an image in the data folder. The model used is VGG19. 

## TODO :
 - [x] Faire fonctionner ce git
 - [x] Utiliser vgg19 pour tout le monde 
 - [x] Utiliser les mêmes images 
 - [ ] Creer des packages ou projet poetry pour chaque méthode 
 - [ ] 

# Grad-Cam
The Grad-Cam method part is managed by Yorick.
This section will regroup a synthesis of what I've done so far, and some explanations on how to use the code for my part.
Credits to [Stepan Ulyanin](https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82) for the Grad-Cam implementation. 

## My part summarised :

 - [x] Read the Grad-Cam article
 - [x] Try to get implemented code for Grad-Cam in python
 - [x] Run the code and **correct** errors
 - [x] Customise the code and make it work with multiple images
 - [x] Create a **working** poetry-ready folder
 - [ ] Find comparison points between each algorithm
 - [ ] Compare each algorithm
 - [ ] *(maybe)* Implement a way to switch between CNNs to add more comparison

## How to use the code

 - With **GitHub** (you need to install all the dependencies **by yourself**) :

Get the [grad-cam folder](https://github.com/LarveTable/etude-pratique-CNNs/tree/21a1fdaff7529f685fdb3f1e3cab8b9a136c9b0b/grad-cam) from the **main** branch

The principal script is `predict_batch.py`, launching this script will try to fetch images in the `data/images` folder, rename and sort them with the script `rename_all_files.py`, then call the script `vgg19_gradcam.py` with each found image.

The `vgg19_gradcam.py` script will apply the **VGG19** model to the image to get a likely prediction and then **pull the gradients** to **generate a heatmap** that will be superimposed on the image. The last step will be to generate a text file that contains **all the predictions** associated with their image.

 If you want to **try it with your own images**, just add them in the *images* folder. 
 You are also able to **delete every files** in the `data/images` and `results` folders using the script `reset_all_files.py`.

- With **Poetry** :

Check that you have **poetry installed** on your system.

Get the [grad-cam folder](https://github.com/LarveTable/etude-pratique-CNNs/tree/21a1fdaff7529f685fdb3f1e3cab8b9a136c9b0b/grad-cam) from the **main** branch

Put the forlder wherever you want and open it with the terminal. Run `poetry install`, it will install all the required dependencies into the poetry virtual environment. If everything went well, you can now run `poetry run python3 grad-cam/predict_batch.py` and see the results.

**WIP**

# SHAP
The Grad-Cam method part is managed by Sulaymane.

## My part summarised :
 - [x] Read the SHAP article
 - [x] Try to run SHAP examples 
 - [x] Use shap with vgg16
 - [x] Change code to use it with vgg19
 - [x] Use shap with vgg19 -> Error : kernel keeps dying for no reason :/ -> fixed (reducing the nsample 
 - [ ] Make the VGG19 work, not in notebook but within a poetry project
 - [ ] Select images remotely fetching from the imagenet database 

## How to use the code

# Integrated Gradient 
The Grad-Cam method part is managed by Sulaymane.

## My part summarised :

 - [x] Read the SHAP article
 - [x] Try to get implemented code for Grad-Cam in python
 - [x] Run the code and **correct** errors
 - [x] Customise the code and make it work with multiple images
 - [ ] Create a **working** package with poetry
 - [ ] Find comparison points between each algorithm
 - [ ] Compare each algorithm
 - [ ] *(maybe)* Implement a way to switch between CNNs to add more comparison

## How to use the code

# LIME 
The LIME part is managed by Adèle.

## My part summarised :

 - [x] Do some research on neural networks
 - [x] Read the LIME article
 - [x] Get implemented codes on github
 - [x] Test the code, correct it
 - [x] Try it with some images
 - [x] Understand the code
 - [x] Try again with different neural networks : VGG19

## How to use the code
