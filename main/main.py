import time
from image_processing import process_dataset
from gradcam.vgg19_gradcam import gradcam_process
import cv2
import os
from tqdm import tqdm
from LIME.VGG19_LIME import lime_process
import matplotlib.pyplot as plt
from Integrated_Gradients.ig import ig_process
from vgg19 import init_model_vgg19

#to comment

def run_comparison(xai_methods, neural_networks, dataset_path):
    if (not xai_methods or not neural_networks or not dataset_path):
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

        # enumerate through the images and apply xai methods
        fbar = tqdm(files)
        for file_name in fbar:
            fbar.set_description("Processing %s" % file_name)
            #print('Processing : '+file_name)
            img, _ = next(iterateur)

            # get the file name without the extension
            file_name_without_extension = os.path.splitext(file_name)[0]
            
            for nn in neural_networks:

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
                            if nn == 'vgg19':
                                selected_nn = selected_nn_gradcam
                                pred_raw = pred_raw_gradcam
                            output_image, time_elapsed, mask, filtered_image = gradcam_process(img, file_name, selected_nn, pred_raw) #mask to intersect and filtered to re inject
                            cv2.imwrite(image_directory+'/'+file_name, output_image)
                            cv2.imwrite(image_directory+'/'+"mask"+file_name, mask)
                            cv2.imwrite(image_directory+'/'+"filtered"+file_name, filtered_image)
                            write_to_file(time_elapsed_directory, file_name_without_extension+'.txt', str(round(time_elapsed, 3))+'s')
                        case 'lime':
                            output_image, time_elapsed, mask, filtered_image = lime_process(img, file_name, selected_nn, pred_raw)
                            cv2.imwrite(image_directory+'/'+file_name, output_image)
                            cv2.imwrite(image_directory+'/'+"mask"+file_name, mask)
                            cv2.imwrite(image_directory+'/'+"filtered"+file_name, filtered_image)
                            write_to_file(time_elapsed_directory, file_name_without_extension+'.txt', str(round(time_elapsed, 3))+'s')
                        case 'shap':
                            #todo
                            cv2.imwrite(image_directory+'/'+file_name, output_image) 
                            pass
                        case 'integrated_gradients':
                            #todo
                            output_image, time_elapsed, mask, filtered_image = ig_process(img, file_name, selected_nn, nn)
                            cv2.imwrite(image_directory+'/'+file_name, output_image)
                            cv2.imwrite(image_directory+'/'+"mask"+file_name, mask)
                            cv2.imwrite(image_directory+'/'+"filtered"+file_name, filtered_image)
                            write_to_file(time_elapsed_directory, file_name_without_extension+'.txt', str(round(time_elapsed, 3))+'s')
                            pass
                        case _:
                            print("Method not found.")
                            return

        global_timer_end = time.time()
        print(f'Work done in {(global_timer_end-global_timer_start):.3}s')

def directories_check(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def write_to_file(directory, file_name, content):
    with open(directory+'/'+file_name, 'w') as file:
        file.write(content)

# for test purposes
run_comparison(["integrated_gradients"], ["vgg19"], 'main/data')
