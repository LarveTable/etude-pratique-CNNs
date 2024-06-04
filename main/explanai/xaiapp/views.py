import time
import json

from django.shortcuts import render, get_list_or_404, get_object_or_404
from django.http import HttpResponseRedirect, StreamingHttpResponse
from django.urls import reverse
from .models import Config, InImage, Experiment,ExplanationMethod, ExplanationResult

import threading
import statistics as stats

from .main import *
from utils.load_coco_images import download

# 
# TODO :
# - Use threading.Event() to create a flag to avoid duplicate threads: 
#       run only if explanation.is_running = false
# - Use pickle to save and get the model easily
#       -> Pickle allows to serialize a class and compress it efficiently 
#            into a binary
# - Use JSON for config files 
# - Use CSRF_token for security in form  
# - check streamlit for deployment
# - check https://www.pythonanywhere.com/ for deployment

# Homepage serving
def home(request):
    '''download()
    # for test purposes
    '''
    return render(request, "xaiapp/home.html")

# render result page with form data and results showing up 
def experiments(request):
    if request.method == 'POST':
        data = request.POST
        images = request.FILES.getlist('images')

        modelName = data['modelName']
        method_selection = data.getlist('methodSelection')

        # save configuration
        new_config_object = Config.objects.create(model_name=modelName)
        new_config_object.save()

        for method_name in method_selection:
            new_config_object.methods.add(ExplanationMethod.objects.get(name=method_name))

        # save images
        for img in images : 
            new_image_object = InImage.objects.create(config=new_config_object,
                                                    image=img)
            new_image_object.save()

        # New experiment
        new_expe_object = Experiment.objects.create(config=new_config_object, status="created")
        new_expe_object.save()

        # Redirect to result/expeID
        return HttpResponseRedirect(reverse("result", args=(new_expe_object.id,)))

    return render(request, "xaiapp/experiments.html")

# display the result page and start explaining if status != finished
# start the processing in the background server
def result(request, experiment_id):
    experiment = get_object_or_404(Experiment, pk=experiment_id)
    config = experiment.config
    # get all method of this configuration
    methods=[m.name for m in list(config.methods.all())]
    images = config.inimage_set.all()

    # if the experiment isn't done then execute its processing in background : 
    # expe is created  then when running then when done its satus is finished
    if experiment.status == "created":
        experiment.status="pending"
        experiment.save()
    
    if experiment.status == "pending":
        try:
            thread1 = threading.Thread(target=process_experiment, args=(experiment_id,))
            thread1.start()
        except Exception as e:
            experiment.status="error"
            experiment.save()
            print(f"Thread Error: {e}")
    
    return render(request, "xaiapp/results.html", {"config_data":config, "in_images":images, "experiment_id":experiment_id, "experiment_status":experiment.status, "methods":methods, "experiment_date":experiment.created_at})

# display all available experiments 1st image, a link too the result page
def experiments_list(request):
    expe_list = [{"id":e.id, "img1":e.config.inimage_set.first(), "date":e.created_at} for e in Experiment.objects.all()]
    return render(request, "xaiapp/experiments_list.html",context={"expe_list":expe_list})

# Result page for an image : in image, prediction, out image for each methods 
# in the experiment, 
def image_result(request, experiment_id, image_id):
    result={}
    lime_m=None
    lime_result=None
    lime_image=None
    lime_mask=None
    lime_time=0
    lime_intersect=0

    integrated_result=None
    integrated_m=None
    integrated_image=None
    integrated_mask=None
    integrated_time=0
    integrated_intersect=0

    gradcam_m=None
    gradcam_result=None
    gradcam_image=None
    gradcam_mask=None
    gradcam_time=0
    gradcam_intersect=0

    # get experiment
    experiment = get_object_or_404(Experiment, pk=experiment_id)

    # get input image 
    in_image=get_object_or_404(InImage, pk=image_id)

    # get explanation result 
    explanation_result = get_object_or_404(ExplanationResult, experiment=experiment, intput_image=in_image)

    # get all methods used for this image
    methods=[m.name for m in list(explanation_result.methods.all())]

    if "lime" in methods:
        lime_m = ExplanationMethod.objects.get(name='lime')
        lime_result=Result.objects.get(intput_image=in_image, method=lime_m)
        lime_image=lime_result.final
        lime_mask=lime_result.mask
        lime_time=round(lime_result.elapsed_time,3)
        lime_intersect=round(next(iter(lime_result.result_intersect.values())),3)

    if "gradcam" in methods:
        gradcam_m = get_object_or_404(ExplanationMethod, name="gradcam")
        gradcam_result=Result.objects.get(intput_image=in_image, method=gradcam_m)
        gradcam_image=gradcam_result.final
        gradcam_mask=gradcam_result.mask
        gradcam_time=round(gradcam_result.elapsed_time,3)
        gradcam_intersect=round(next(iter(gradcam_result.result_intersect.values())),3)

    if "integrated_gradients" in methods:
        integrated_m = get_object_or_404(ExplanationMethod, name="integrated_gradients")
        integrated_result=Result.objects.get(intput_image=in_image, method=integrated_m)
        integrated_image=integrated_result.final
        integrated_mask=integrated_result.mask
        integrated_time=round(integrated_result.elapsed_time,3)
        integrated_intersect=round(next(iter(integrated_result.result_intersect.values())),3)

    # if COCO : get Coco masks 
    if gradcam_result:
        coco_mask=gradcam_result.coco_masks
    elif lime_result:
        coco_mask=lime_result.coco_masks
    elif integrated_result:
        coco_mask=integrated_result.coco_masks

    print(str(in_image))   
    total_time=round((lime_time + gradcam_time + integrated_time), 3)
    if in_image.status == "finished":
        result={
                "lime_image":lime_image,
                "integrated_image":integrated_image,
                "gradcam_image":gradcam_image,
                "coco_mask":coco_mask,
                "gradcam_mask":gradcam_mask,
                "integrated_mask":integrated_mask,
                "lime_mask":lime_mask,
                "prediction":explanation_result.pred_top1,
                "gradcam_time":str(gradcam_time),
                "lime_time":str(lime_time),
                "integrated_time":str(integrated_time),
                "total_time":total_time,
                "gradcam_intersect":gradcam_intersect,
                "lime_intersect":lime_intersect,
                "integrated_intersect":integrated_intersect,
                }
        return render(request, "xaiapp/image_result.html", {"in_image":in_image, "img_name":str(in_image), "experiment_id":experiment_id, "result":result})
        
    return render(request, "xaiapp/image_result.html", {"in_image":in_image, "img_name":str(in_image), "experiment_id":experiment_id})

def cred(request):
    return render(request, "xaiapp/cred.html")

# process each inimage according to config
def process_experiment(experiment_id):
    # get experiment at this id
    experiment = get_object_or_404(Experiment, pk=experiment_id)
    # get configuration 
    config = experiment.config
    # parameters fo future implementation
    parameters = {m.name :{} for m in list(config.methods.all())}
    # get all method of this configuration
    methods=[m.name for m in list(config.methods.all())]

    try:
        exp = run_comparison(methods, [config.model_name], parameters, experiment_id, True, ['dog'])
    except Exception as e:
        experiment.status="error"
        experiment.save()
        print(f"Error when running: {e}")
    # All results in exp object
    print(exp.results)

    ''' fake process for demo
    experiment = get_object_or_404(Experiment, pk=experiment_id)
    config = experiment.config
    for iimg in config.inimage_set.all():
        time.sleep(10) # simulate delay for testing
        iimg.status = "finished"
        print("finished : ", str(iimg.image))
        iimg.save()
    experiment.status = "finished"
    experiment.save()
    '''

# Update the experiment data : images and experiment status
# todo : restructure data : image->time,status; all-time, min..., status
def get_experiment_update(request, experiment_id):
    # get experiment
    experiment = get_object_or_404(Experiment, pk=experiment_id)

    # get config 
    config=experiment.config
    methods=[m.name for m in list(config.methods.all())]

    def event_stream():
        while 1:
            time.sleep(1)
            print("status:", experiment.status)
            # each image : status and if done out image
            data = { 
                "message": "Experiment update",
                "experiment":{
                    "date":str(experiment),
                    "status":experiment.status,
                },
                "status":[]
            }
            img_time_array=[]
            gradcam_img_time_array=[]
            lime_img_time_array=[]
            integrated_img_time_array=[]
            for iimg in config.inimage_set.all():
                # total time with this image (all methods)
                img_time=0
                method_time=0
                for m in methods:
                    # time for this image with method m
                    if iimg.status == "finished":
                        method_time=round(Result.objects.get(method=ExplanationMethod.objects.get(name=m), intput_image=iimg).elapsed_time,4)
                    
                    img_time+=method_time
                    match m: 
                        case "gradcam":
                            gradcam_img_time_array.append(method_time)
                        case "lime":
                            lime_img_time_array.append(method_time)
                        case "integrated_gradients":
                            integrated_img_time_array.append(method_time)

                img_time_array.append(img_time)

                iimg_status = {"imgName":str(iimg.image), "status": iimg.status, "id":iimg.id, "img_time":round(img_time,3)}
                data["status"].append(iimg_status)

            # stats:
            expe_statistics = {
                "total_time":round(sum(img_time_array),3),
                "max_time":round(max(img_time_array),3),
                "min_time":round(min(img_time_array),3),
                "mean_time":round(stats.mean(img_time_array),3),
                "median":round(stats.median(img_time_array),3),
            }
            if len(img_time_array) > 1:
                expe_statistics['variance'] = round(stats.variance(img_time_array))

            for m in methods:
                match m: 
                    case "gradcam":
                        expe_statistics['gradcam'] = {
                            "total_time":round(sum(gradcam_img_time_array),3),
                            "max_time":round(max(gradcam_img_time_array),3),
                            "min_time":round(min(gradcam_img_time_array),3),
                            "mean_time":round(stats.mean(gradcam_img_time_array),3),
                            "median":round(stats.median(gradcam_img_time_array),3),
                        }
                        if len(gradcam_img_time_array) > 1:
                            expe_statistics['gradcam']['variance'] = round(stats.variance(gradcam_img_time_array),3)
                    case "lime":
                        expe_statistics['lime'] = {
                            "total_time":round(sum(lime_img_time_array),3),
                            "max_time":round(max(lime_img_time_array),3),
                            "min_time":round(min(lime_img_time_array),3),
                            "mean_time":round(stats.mean(lime_img_time_array),3),
                            "median":round(stats.median(lime_img_time_array),3),
                        }
                        if len(lime_img_time_array) > 1:
                            expe_statistics['lime']['variance'] = round(stats.variance(lime_img_time_array),3)
                    case "integrated_gradients":
                        expe_statistics['integrated_gradients'] = {
                            "total_time":round(sum(integrated_img_time_array),3),
                            "max_time":round(max(integrated_img_time_array),3),
                            "min_time":round(min(integrated_img_time_array),3),
                            "mean_time":round(stats.mean(integrated_img_time_array),3),
                            "median":round(stats.median(integrated_img_time_array),3),
                        }
                        if len(integrated_img_time_array) > 1:
                            expe_statistics['integrated_gradients']['variance'] = round(stats.variance(integrated_img_time_array),3)

            data["statistics"]=expe_statistics
            data["methods"]=methods
            json_data = json.dumps(data)
            yield f"data: {json_data}\n\n"

    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    return response



def documentation(request):
    return render(request, "xaiapp/documentation.html")