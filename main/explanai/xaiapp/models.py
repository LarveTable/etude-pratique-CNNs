from django.db import models

# TODO:
# - Change : Image -> InputImage
# - New: Experiment : #config_id, status:enum, 
# use choices for defined models

class ExplanationMethod(models.Model):
    name = models.CharField(max_length=30)

    def __str__(self):
        return self.name

class Config(models.Model):
    model_name = models.CharField(max_length=50)
    methods=models.ManyToManyField(ExplanationMethod)

    def __str__(self) -> str:
        return str(self.id)


class InImage(models.Model):
    config = models.ForeignKey(Config, on_delete=models.CASCADE)
    status = models.CharField(max_length=200,default="pending")
    image = models.ImageField(null=False, blank=False, upload_to="input_images/")

# A config that's been executed containing results with images
class Experiment(models.Model):
    config = models.ForeignKey(Config, on_delete=models.CASCADE)
    status = models.CharField(max_length=200)

class Result(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    status = models.BooleanField(default=False) # finish

class OutImage(models.Model):
    input_image = models.ForeignKey(InImage, on_delete=models.CASCADE)
    image = models.ImageField(null=False, blank=False, upload_to="output_images/")
class Stat(models.Model):
    result = models.ForeignKey(Result, on_delete=models.CASCADE)
    time = models.IntegerField(default=0)