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

#to comment

def run_comparison(xai_methods, neural_networks, parameters, dataset_path=None, use_coco=False, coco_categories=None
                   , number_of_images=None, randomized=None):
    if (not xai_methods or not neural_networks or not dataset_path or not use_coco):
        print("At least one parameter is missing.")
    else:
        print("All parameters are present, processing...")
        # start the global timer
        global_timer_start = time.time()

        # start the dataset processing timer
        dataset_process_timer_start = time.time()

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

            # enumerate through the images and apply xai methods
            fbar = tqdm(files)
            for file_name in fbar:
                fbar.set_description("Processing %s" % file_name)
                #print('Processing : '+file_name)
                img, _ = next(iterateur)

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
                        write_to_file(preds_directory, file_name_without_extension+'.txt'+'_top1', pred_top1)
                        write_to_file(preds_directory, file_name_without_extension+'.txt'+'_top5', preds_top5)

                for method in xai_methods:

                    # check if the image directory exists, if not create it
                    image_directory = 'main/results/images/'+method+'/'+method+'_'+file_name_without_extension
                    time_elapsed_directory = 'main/results/times/'+method+'/'+method+'_'+file_name_without_extension
                    directories_check([image_directory, time_elapsed_directory])

                    match method:
                        case 'gradcam':
                            output_image, time_elapsed, mask, filtered_image, affected_pixels_method = gradcam_process(img, file_name, selected_nn_gradcam, pred_raw_gradcam, parameters['gradcam']) #mask to intersect and filtered to re inject
                            cv2.imwrite(image_directory+'/'+file_name, output_image)
                            cv2.imwrite(image_directory+'/'+"mask"+file_name, mask)
                            cv2.imwrite(image_directory+'/'+"filtered"+file_name, filtered_image)
                            write_to_file(time_elapsed_directory, file_name_without_extension+'.txt', str(round(time_elapsed, 3))+'s')
                            result_intersect = evaluate(int(id[0]), coco_categories, mask)
                            processed_filter = process_one(filtered_image)
                            _, _, second_pass_pred, _ = init_model_vgg19(processed_filter, file_name, use_gradcam=False) #vgg19 prend en compte les pixels noirs, on les rend transparents ?
                            explanation.save_result(method, output_image, mask, filtered_image, time_elapsed, pred_top1, 
                                                    preds_top5, second_pass_pred, result_intersect, int(id[0]), use_coco, coco_categories)
                        case 'lime':
                            output_image, time_elapsed, mask, filtered_image = lime_process(img, file_name, selected_nn, pred_raw, parameters['lime'])
                            cv2.imwrite(image_directory+'/'+file_name, output_image)
                            cv2.imwrite(image_directory+'/'+"mask"+file_name, mask)
                            cv2.imwrite(image_directory+'/'+"filtered"+file_name, filtered_image)
                            write_to_file(time_elapsed_directory, file_name_without_extension+'.txt', str(round(time_elapsed, 3))+'s')
                            result_intersect = evaluate(int(id[0]), coco_categories, mask)
                            processed_filter = process_one(filtered_image)
                            _, _, second_pass_pred, _ = init_model_vgg19(processed_filter, file_name, use_gradcam=False)
                            explanation.save_result(method, output_image, mask, filtered_image, time_elapsed, pred_top1, 
                                                    preds_top5, second_pass_pred, result_intersect, int(id[0]), use_coco, coco_categories)
                        case 'shap':
                            #todo
                            cv2.imwrite(image_directory+'/'+file_name, output_image) 
                            pass
                        case 'integrated_gradients':
                            #todo
                            output_image, time_elapsed, mask, filtered_image = ig_process(img, file_name, selected_nn, nn, parameters['integrated_gradients'])
                            cv2.imwrite(image_directory+'/'+file_name, output_image)
                            cv2.imwrite(image_directory+'/'+"mask"+file_name, mask)
                            cv2.imwrite(image_directory+'/'+"filtered"+file_name, filtered_image)
                            write_to_file(time_elapsed_directory, file_name_without_extension+'.txt', str(round(time_elapsed, 3))+'s')
                            result_intersect = evaluate(int(id[0]), coco_categories, mask)
                            processed_filter = process_one(filtered_image)
                            _, _, second_pass_pred, _ = init_model_vgg19(processed_filter, file_name, use_gradcam=False)
                            explanation.save_result(method, output_image, mask, filtered_image, time_elapsed, pred_top1, 
                                                    preds_top5, second_pass_pred, result_intersect, int(id[0]), use_coco, coco_categories)
                        case _:
                            print("Error : method not found.")
                            return

        global_timer_end = time.time()
        print(f'Work done in {(global_timer_end-global_timer_start):.3}s')
        return explanation

def directories_check(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def write_to_file(directory, file_name, content):
    with open(directory+'/'+file_name, 'w') as file:
        file.write(content)

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

    exp = run_comparison(["gradcam"], ["vgg19"], parameters, 'main/data', True, ['dog'])

    print(exp.results['gradcam'])
    #filter = exp.results['gradcam'][161609]['filtered_image']
    #cv2.imshow('filtered', filter)
    #cv2.waitKey(0)
    #print(exp.results['lime'][90003]['result_intersect'])
