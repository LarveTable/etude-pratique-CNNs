# etude-pratique-CNNs
> Repository to share our codes and other useful files on our CNNs
> explanation project. By Adèle, Sulaymane, Taha, Yorick.

# Grad-Cam
The Grad-Cam method part is managed by Yorick.
This section will regroup a synthesis of what I've done so far, and some explanations on how to use the code for my part.

## My part summarised :

 - [x] Read the Grad-Cam article
 - [x] Try to get implemented code for Grad-Cam in python
 - [x] Run the code and **correct** errors
 - [x] Customise the code and make it work with multiple images
 - [ ] Create a **working** package with poetry
 - [ ] Find comparison points between each algorithm
 - [ ] Compare each algorithm
 - [ ] *(maybe)* Implement a way to switch between CNNs to add more comparison

## How to use the code

 - With GitHub :

Get the [grad-cam folder](https://github.com/LarveTable/etude-pratique-CNNs/tree/21a1fdaff7529f685fdb3f1e3cab8b9a136c9b0b/grad-cam) from the **main** branch

The principal script is `predict_batch.py`, launching this script will try to fetch images in the `data/images` folder, rename and sort them with the script `rename_all_files.py`, then call the script `vgg19_gradcam.py` with each found image.

The `vgg19_gradcam.py` script will apply the **VGG19** model to the image to get a likely prediction and then **pull the gradients** to **generate a heatmap** that will be superimposed on the image. The last step will be to generate a text file that contains **all the predictions** associated with their image.

 If you want to **try it with your own images**, just add them in the *images* folder. 
 You are also able to **delete every files** in the `data/images` and `results` folders using the script `reset_all_files.py`.

- ~~With Poetry :~~

**WIP**

# SHAP
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
 - [ ] Undertand the code
 - [ ] Try again with different neural networks  

## How to use the code
