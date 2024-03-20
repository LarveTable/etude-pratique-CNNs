import time
from image_processing import process_dataset
from gradcam.vgg19_gradcam import gradcam_process

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
        print(f'Dataset processed in {dataset_process_timer_end-dataset_process_timer_start:.3}ms')

        # enumerate through the images and apply xai methods
        for file_name in files:
            print('Processing : '+file_name)
            img, _ = next(iterateur)

            for method in xai_methods:
                for nn in neural_networks:
                    print(f'Processing {method} with {nn}')
                    # each method should be a function that takes an image, a file name and NNs as parameters
                    # return must be a processed image, {a prediction?} and a time elapsed and maybe more
                    match method:
                        case 'gradcam':
                            gradcam_process(img, file_name, nn)
                        case 'lime':
                            pass
                        case 'shap':
                            pass
                        case 'integrated_gradients':
                            # also process baseline
                            pass

        global_timer_end = time.time()
        print(f'Work done in {global_timer_end-global_timer_start:.3}ms')

run_comparison([], [], [])