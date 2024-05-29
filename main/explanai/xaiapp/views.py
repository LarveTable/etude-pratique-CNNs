import time
import json

from django.shortcuts import render, get_list_or_404, get_object_or_404
from django.http import HttpResponseRedirect, StreamingHttpResponse
from django.urls import reverse
from .models import Config, InImage, Experiment

import threading

from .main import *
# 
# TODO :
# - Use pickle to save and get the model easily
#       -> Pickle allows to serialize a class and compress it efficiently 
#            into a binary
# - Find a way to work with images maybe pickle again
# - Use JSON for config files 
# - Use CSRF_token for security in form  
# - check streamlit for deployment
# - check https://www.pythonanywhere.com/ for deployment
# - create a requestExperiment class containing all the data necessary
# - process_request method to call all algorithm and gives the output as a experiment output 
# - sub_process to get image by images 
# - use django forms
# - transform images to 224x224 before saving

# Homepage serving
def home(request):
    '''# for test purposes
    parameters = {
        "gradcam": {
        },
        "lime": {
        },
        "integrated_gradients": {
        }
    }

    exp = run_comparison(["gradcam"], ["vgg19"], parameters, 'main/data', True, ['dog'])

    print(exp.results['gradcam'])'''
    return render(request, "xaiapp/home.html")

# render result page with form data and results showing up 
def experiments(request):
    if request.method == 'POST':
        data = request.POST
        images = request.FILES.getlist('images')

        modelName = data['modelName']

        # save configuration
        new_config_object = Config.objects.create(model_name=modelName)
        new_config_object.save()

        # save images
        for img in images : 
            new_image_object = InImage.objects.create(config=new_config_object,
                                                    image=img)
            new_image_object.save()

        # New experiment
        new_expe_object = Experiment.objects.create(config=new_config_object, status="pending")
        new_expe_object.save()

        # Redirect to result/expeID
        return HttpResponseRedirect(reverse("result", args=(new_expe_object.id,)))

    return render(request, "xaiapp/experiments.html")

# display the result page and start explaining if status != finished
# start the processing in the background server
def result(request, experiment_id):
    experiment = get_object_or_404(Experiment, pk=experiment_id)
    config = experiment.config
    images = config.inimage_set.all()

    # if the experiment isn't done then execute its processing in background
    if experiment.status != "finished":
        thread1 = threading.Thread(target=process_experiment, args=(experiment_id,))
        thread1.start()
    
    return render(request, "xaiapp/results.html", {"config_data":config, "in_images":images, "experiment_id":experiment_id, "experiment_status":experiment.status})

# process each inimage and put its out image 
def process_experiment(experiment_id):
    print("processing")
    experiment = get_object_or_404(Experiment, pk=experiment_id)
    config = experiment.config
    for iimg in config.inimage_set.all():
        time.sleep(2) # simulate delay for testing
        iimg.status = "finished"
        print("finished : ", str(iimg.image))
        iimg.save()
    experiment.status = "finished"
    experiment.save()

# Update the experiment data : images and its status
def get_experiment_update(request, experiment_id):
    experiment = get_object_or_404(Experiment, pk=experiment_id)
    config=experiment.config
    print("status = " , experiment.status)
    def event_stream():
        while 1:
            time.sleep(1)
            # each image : status and if done out image
            data = { 
                "message": "Experiment update",
                "status":[]
            }
            for iimg in config.inimage_set.all():
                # name, status, outimage
                iimg_status = {"imgName":str(iimg.image), "status": iimg.status}
                data["status"].append(iimg_status)

            print(data)
            json_data = json.dumps(data)
            yield f"data: {json_data}\n\n"

    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    return response