# Explainable AI: A Comparison Web Application
By Adèle, Sulaymane, Taha, Yorick.

### Summary
This is the code of a practical study "A Comparative Study of Neural Network Explanation Methods" realized in 2023/2024 by four INSA students. During this study,  we have created a web application explaining what machine learning classifiers (models) are doing under the hood of a prediction. It allows you to compare XAI methods upon a a group of metrics. 

This application is meant to accelerate the process of choosing the right explanation method for a certain CNN model.  You can for example import some images from the [COCO dataset](https://cocodataset.org/) and see how close the explanation of a prediction is to the real zone of interest of the image. 
The work done heare is intended to be extended, adding models, methods and other insights can be done easily in the back-end and front-end.

### Dependencies 

### Tutorial 
1. Clone project
2. Install all dependencies with pip
3. Place the coco dataset in the right folder and follow the instructions
4. You can now start the server with python3 manage.py runserver
5. On the opened webpage, you can start a new experiment

### Experimenting 
1. Configure your experiment by chosing a model, a group of image, the explanation methods you want to compare and click start
2. You will be able to see all images being processed and clicking an image will open its comparison page where you can check all explanation from different methods.
3. You can save the link to the explanation and come back later or share it with others. 

### Credits & references
[Stepan Ulyanin](https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82) 
[Grad-CAM](https://arxiv.org/pdf/1610.02391.pdf)
[SHAP](https://github.com/shap/shap)
[LIME](https://github.com/marcotcr/lime)
[Integrated Gradients](https://github.com/ankurtaly/Integrated-Gradients)
