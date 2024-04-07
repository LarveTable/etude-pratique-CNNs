from django.shortcuts import render, HttpResponse
from .models import Config, Image
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

# Create your views here.
def home(request):
    return render(request, "home.html")

# render result page with form data and results showing up 
def experiments(request):
    if request.method == 'POST':
        data = request.POST
        images = request.FILES.getlist('images')

        print(request.FILES)
        modelName = data['modelName']

        # save configuration
        new_config_object = Config.objects.create(model_name=modelName)
        new_config_object.save()

        # save images
        for img in images : 
            new_image_object = Image.objects.create(config=new_config_object,
                                                    image=img)
            new_image_object.save()

        # if no error 
        context={}
        context['config-data'] = "hi"
        return render(request, "results.html", context)

    return render(request, "experiments.html")

