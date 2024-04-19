from django.shortcuts import render, get_list_or_404, get_object_or_404
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from .models import Config, Image, Experiment
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

# Create your views here.
def home(request):
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
            new_image_object = Image.objects.create(config=new_config_object,
                                                    image=img)
            new_image_object.save()

        # New experiment
        new_expe_object = Experiment.objects.create(config=new_config_object)
        new_expe_object.save()

        # if no error 
        context={}
        context['config_data'] = "hi"

        # Redirect to result/expeID
        return HttpResponseRedirect(reverse("result", args=(new_expe_object.id,)))

    return render(request, "xaiapp/experiments.html")

# run in bg and update as the results come 
def result(request, experiment_id):
    experiment = get_object_or_404(Experiment, pk=experiment_id)
    config = experiment.config
    images = config.image_set.all()
    
    for img in images:
        print(img.image)
    # return HttpResponse("ok")
    return render(request, "xaiapp/results.html", {"config_data":config})