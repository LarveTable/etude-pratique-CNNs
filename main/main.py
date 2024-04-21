import time
from image_processing import process_dataset
from gradcam.vgg19_gradcam import gradcam_process
import cv2
import os
from tqdm import tqdm

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
            for method in xai_methods:
                for nn in neural_networks:
                    #print(f'Processing {method} with {nn}')
                    # each method should be a function that takes an image, a file name and NNs as parameters
                    # return must be a processed image, {a prediction?} and a time elapsed and maybe more

                    # check if the image directory exists, if not create it
                    file_name_without_extension = os.path.splitext(file_name)[0]
                    image_directory = 'main/results/images/'+method+'/'+method+'_'+file_name_without_extension
                    preds_directory = 'main/results/predictions/'+method+'/'+method+'_'+file_name_without_extension
                    time_elapsed_directory = 'main/results/times/'+method+'/'+method+'_'+file_name_without_extension
                    directories_check([image_directory, preds_directory, time_elapsed_directory])

                    match method:
                        case 'gradcam':
                            output_image, preds, time_elapsed, mask, inversed_image = gradcam_process(img, file_name, nn)
                            cv2.imwrite(image_directory+'/'+file_name, output_image)
                            write_to_file(preds_directory, file_name_without_extension+'.txt', preds)
                            write_to_file(time_elapsed_directory, file_name_without_extension+'.txt', str(round(time_elapsed, 3))+'s')
                        case 'lime':
                            #todo
                            cv2.imwrite(image_directory+'/'+file_name, output_image) 
                            lime_vgg = LimeVGG19()
                            img = lime_vgg.get_image('./Trousse.jpg')
                            # Obtenir les prédictions top-5
                            predictions_top5 = lime_vgg.predict(img)
                            # on demande d'expliquer la prédiciton de l'image dont le path est passé en paramètre : 
                            explanation = lime_vgg.explain_image('./Trousse.jpg')
                            # on demande de visualiser l'explication de la prédiction via Lime 
                            lime_vgg.visualize_explanation(explanation, './Trousse.jpg')
                        case 'shap':
                            #todo
                            cv2.imwrite(image_directory+'/'+file_name, output_image) 
                            pass
                        case 'integrated_gradients':
                            #todo
                            # also process baseline
                            cv2.imwrite(image_directory+'/'+file_name, output_image)
                            pass

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
run_comparison(["gradcam"], ["vgg19"], 'main/data')
