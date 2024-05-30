import time
from utils.image_processing import process_dataset, process_one
from methods.gradcam.vgg19_gradcam import gradcam_process
import cv2
import os
from tqdm import tqdm
from methods.LIME.VGG19_LIME import lime_process
import matplotlib.pyplot as plt
from methods.Integrated_Gradients.ig import ig_process
from utils.vgg19 import init_model_vgg19
from utils.evaluate_intersection import evaluate
import re
from utils.explanation import Explanation
import random
import shutil
from datetime import datetime
from PIL import Image
import tempfile
from django.core.files import File
import numpy as np

#BDD
from django.shortcuts import render, get_list_or_404, get_object_or_404
from django.http import HttpResponseRedirect, StreamingHttpResponse
from django.urls import reverse
from .models import Result, OutImage, Experiment, ExplanationMethod, CocoCategories

#to comment
#todo
#close the open
#do the same for the other methods
#remirgate db

def run_comparison(xai_methods, neural_networks, parameters, expe_id, use_coco=False, coco_categories=None):
    if (not xai_methods or not neural_networks or not parameters or not expe_id):
        print("At least one parameter is missing.")
    elif (use_coco and not coco_categories):
        print("COCO categories are missing.")
    else:
        print("All parameters are present, processing...")
        # start the global timer
        global_timer_start = time.time()

        # get date in YYYYMMDD format
        date = datetime.today().strftime('%Y-%m-%d')

        # start the dataset processing timer
        dataset_process_timer_start = time.time()

        # create a tmp file to store the dataset
        dataset_path = "./main/tmp"+str(expe_id)
        directories_check([dataset_path+"/data"])

        experiment = get_object_or_404(Experiment, pk=expe_id)
        config = experiment.config
        for iimg in config.inimage_set.all():
            img_path = "main/explanai/media/"+str(iimg.image)
            img = cv2.imread(img_path)
            cv2.imwrite(dataset_path+'/data/'+os.path.basename(img_path), img)

        print("Processing dataset...")
        # process the dataset
        iterateur, files = process_dataset(dataset_path)

        if (not iterateur or not files):
            print("Error processing the dataset.")
            return
        dataset_process_timer_end = time.time()
        print(f'Dataset processed in {(dataset_process_timer_end-dataset_process_timer_start):.3}s\n')
        
        for nn in neural_networks:

            rand_int = random.randint(0, 100)
            time_now = time.time()
                    
            explanation = Explanation(int(time_now*rand_int), xai_methods, nn, parameters)

            # get experiment at this id
            experiment = get_object_or_404(Experiment, pk=expe_id)
            # get configuration 
            config = experiment.config

            # enumerate through the images and apply xai methods
            #fbar = tqdm(files)
            #for file_name in fbar:
            for iimg in tqdm(config.inimage_set.all()):
                #fbar.set_description("Processing %s" % file_name)
                #print('Processing : '+file_name)
                img, _ = next(iterateur)

                file_name = os.path.basename(str(iimg.image))

                file_name_without_extension = file_name.rsplit('_', 1)[0] # get rid of the string added by django

                # get the file name without the extension
                file_name_without_extension = os.path.splitext(file_name)[0]

                # create a regex pattern for the id
                id_pattern = r"^(\d*)"

                # apply the regex pattern to the file name without extension
                id = re.findall(id_pattern, file_name_without_extension) #id[0] contains the id

                # instantiate the neural network and return the model and the predictions
                match nn:
                    case 'vgg19':
                        selected_nn, preds_top5, pred_top1, pred_raw = init_model_vgg19(img, file_name, use_gradcam=False)
                        selected_nn_gradcam, _, _, pred_raw_gradcam = init_model_vgg19(img, file_name, use_gradcam=True)
                        preds_directory = 'main/results/predictions/'+nn+'/'+file_name_without_extension
                        directories_check([preds_directory])
                        write_to_file(preds_directory, file_name_without_extension+'_top1.txt', pred_top1)
                        write_to_file(preds_directory, file_name_without_extension+'_top5.txt', preds_top5)

                for method in xai_methods:

                    # check if the image directory exists, if not create it
                    image_directory = 'main/results/images/'+method+'/'+method+'_'+file_name_without_extension
                    time_elapsed_directory = 'main/results/times/'+method+'/'+method+'_'+file_name_without_extension
                    directories_check([image_directory, time_elapsed_directory])

                    match method:
                        case 'gradcam':
                            output_image, time_elapsed, mask, filtered_image, affected_pixels_method = gradcam_process(img, file_name, selected_nn_gradcam, pred_raw_gradcam, dataset_path, parameters['gradcam']) #mask to intersect and filtered to re inject
                            
                            # Scale the pixel values to the range [0, 255]
                            output_image = (output_image - np.min(output_image)) / (np.max(output_image) - np.min(output_image)) * 255

                            # Convert the data type to uint8
                            output_image = output_image.astype(np.uint8)
                            
                            cv2.imwrite(image_directory+'/'+file_name, output_image)
                            cv2.imwrite(image_directory+'/'+"mask"+file_name, mask)
                            cv2.imwrite(image_directory+'/'+"filtered"+file_name, filtered_image)
                            write_to_file(time_elapsed_directory, file_name_without_extension+'.txt', str(round(time_elapsed, 3))+'s')
                            result_intersect, coco_masks = evaluate(int(id[0]), coco_categories, mask)
                            cv2.imwrite(image_directory+'/'+"coco_masks"+file_name, coco_masks)
                            processed_filter = process_one(filtered_image)
                            _, _, second_pass_pred, _ = init_model_vgg19(processed_filter, file_name, use_gradcam=False) #vgg19 prend en compte les pixels noirs, on les rend transparents ?
                            explanation.save_result(method, output_image, mask, filtered_image, time_elapsed, pred_top1, 
                                                    preds_top5, second_pass_pred, result_intersect, int(id[0]), coco_masks, use_coco, coco_categories)
                            
                            # Obtenez ou créez les instances de CocoCategories
                            coco_categories_instances = []
                            for name in coco_categories:
                                category, _ = CocoCategories.objects.get_or_create(name=name)
                                coco_categories_instances.append(category)

                            ex_res = experiment.explanationresult_set.create(neural_network=nn, date=date)
                            for method_name in xai_methods:
                                ex_res.methods.add(ExplanationMethod.objects.get(name=method_name))  # Utilisation de la méthode set() pour les ManyToMany
                            ex_res.save()

                            res = ex_res.result_set.create(elapsed_time=time_elapsed, pred_top1=pred_top1, second_pass_pred=second_pass_pred, result_intersect=result_intersect, use_coco=use_coco)
                            res.coco_categories.set(coco_categories_instances)
                            res.save()

                            filtered_image = numpy_array_to_django_file(filtered_image, dataset_path+"/out")
                            output_image = numpy_array_to_django_file(output_image, dataset_path+"/out")
                            mask = numpy_array_to_django_file(mask, dataset_path+"/out")
                            coco_masks = numpy_array_to_django_file(coco_masks, dataset_path+"/out")

                            method_instance = ExplanationMethod.objects.get(name=method)
                            out_img = res.outimage_set.create(method=method_instance, final=output_image, mask=mask, filtered=filtered_image, coco_masks=coco_masks)
                            out_img.save()

                            #retrieve elapsed time from db
                            #elapsed_time = Result.objects.get(explanation_results=ex_res).elapsed_time
                            #print("elapsed time : ", elapsed_time)

                        case 'lime':
                            output_image, time_elapsed, mask, filtered_image = lime_process(img, file_name, selected_nn, pred_raw, dataset_path, parameters['lime'])
                            cv2.imwrite(image_directory+'/'+file_name, output_image)
                            cv2.imwrite(image_directory+'/'+"mask"+file_name, mask)
                            cv2.imwrite(image_directory+'/'+"filtered"+file_name, filtered_image)
                            write_to_file(time_elapsed_directory, file_name_without_extension+'.txt', str(round(time_elapsed, 3))+'s')
                            result_intersect, coco_masks = evaluate(int(id[0]), coco_categories, mask)
                            cv2.imwrite(image_directory+'/'+"coco_masks"+file_name, coco_masks)
                            processed_filter = process_one(filtered_image)
                            _, _, second_pass_pred, _ = init_model_vgg19(processed_filter, file_name, use_gradcam=False)
                            explanation.save_result(method, output_image, mask, filtered_image, time_elapsed, pred_top1, 
                                                    preds_top5, second_pass_pred, result_intersect, int(id[0]), coco_masks, use_coco, coco_categories)
                        case 'shap':
                            #todo
                            cv2.imwrite(image_directory+'/'+file_name, output_image) 
                            pass
                        case 'integrated_gradients':
                            #todo
                            output_image, time_elapsed, mask, filtered_image = ig_process(img, file_name, selected_nn, nn, dataset_path, parameters['integrated_gradients'])
                            cv2.imwrite(image_directory+'/'+file_name, output_image)
                            cv2.imwrite(image_directory+'/'+"mask"+file_name, mask)
                            cv2.imwrite(image_directory+'/'+"filtered"+file_name, filtered_image)
                            write_to_file(time_elapsed_directory, file_name_without_extension+'.txt', str(round(time_elapsed, 3))+'s')
                            result_intersect, coco_masks = evaluate(int(id[0]), coco_categories, mask)
                            cv2.imwrite(image_directory+'/'+"coco_masks"+file_name, coco_masks)
                            processed_filter = process_one(filtered_image)
                            _, _, second_pass_pred, _ = init_model_vgg19(processed_filter, file_name, use_gradcam=False)
                            explanation.save_result(method, output_image, mask, filtered_image, time_elapsed, pred_top1, 
                                                    preds_top5, second_pass_pred, result_intersect, int(id[0]), coco_masks, use_coco, coco_categories)
                        case _:
                            print("Error : method not found.")
                            return
            iimg.status = "finished"
            print("finished : ", str(iimg.image))
            iimg.save()
        
        experiment.status = "finished"
        experiment.save()

        global_timer_end = time.time()
        print(f'Work done in {(global_timer_end-global_timer_start):.3}s')
        delete_directory(dataset_path)
        return explanation

def directories_check(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def write_to_file(directory, file_name, content):
    with open(directory+'/'+file_name, 'w') as file:
        file.write(content)

def delete_directory(directory):
    shutil.rmtree(directory)

def numpy_array_to_django_file(image_array, save_dir):
    """
    Convert a NumPy array representing an image to a Django File object.

    Args:
    - image_array: The NumPy array representing the image.
    - save_dir: Directory where the temporary file should be created.

    Returns:
    - File: The Django File object containing the image data.
    """
    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Ensure the NumPy array is of type uint8 and values are in the range 0-255
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)
    if image_array.max() > 255 or image_array.min() < 0:
        image_array = np.clip(image_array, 0, 255)

    # Convert the NumPy array to a PIL image
    image_pil = Image.fromarray(image_array)

    # Determine the file format based on the original image format
    file_format = 'JPEG' if image_pil.mode in ('RGB', 'RGBA') else 'PNG'
    suffix = '.' + file_format.lower()

    # Create a temporary file in the specified directory and write the image to it
    temp_file_path = os.path.join(save_dir, next(tempfile._get_candidate_names()) + suffix)
    image_pil.save(temp_file_path, format=file_format)

    # Open the file and create a Django File object
    temp_file = open(temp_file_path, 'rb')
    django_file = File(temp_file, name=os.path.basename(temp_file_path))
    
    return django_file

if __name__ == "__main__":
    # for test purposes
    parameters = {
        "gradcam": {
        },
        "lime": {
        },
        "integrated_gradients": {
        }
    }

    exp = run_comparison(["gradcam"], ["vgg19"], parameters, None, True, ['dog'])

    print(exp.results['gradcam'])
    #filter = exp.results['gradcam'][161609]['filtered_image']
    #cv2.imshow('filtered', filter)
    #cv2.waitKey(0)
    #print(exp.results['lime'][90003]['result_intersect'])
