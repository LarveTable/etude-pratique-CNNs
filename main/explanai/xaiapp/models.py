import os
from django.db import models
from django.utils import timezone

# TODO:
# - Change : Image -> InputImage
# - New: Experiment : #config_id, status:enum, 
# use choices for defined models

class ExplanationMethod(models.Model):
    class Method(models.TextChoices):
            LM="1","lime"
            GC="2","gradcam"
            IG="3","integrated_gradients"
            SH="4","shap"
    name = models.CharField(max_length=30, choices=Method)

    def __str__(self):
        return self.name

class CocoCategories(models.Model):
    name = models.CharField(max_length=30)

    def __str__(self):
        return self.name

# Configuration
class Config(models.Model):
    model_name = models.CharField(max_length=50)
    methods=models.ManyToManyField(ExplanationMethod)
    use_coco = models.BooleanField(null=False, blank=False, default=False)
    coco_categories = models.ManyToManyField(CocoCategories)

    def __str__(self) -> str:
        return str(self.id)

# Input image
class InImage(models.Model):
    config = models.ForeignKey(Config, on_delete=models.CASCADE)
    status = models.CharField(max_length=200,default="pending")
    image = models.ImageField(null=False, blank=False, upload_to="input_images/")
    def __str__(self) -> str:
        file_name = os.path.basename(str(self.image.url))
        file_name_without_extension = file_name.rsplit('_', 1)[0] # get rid of the string added by django
        # get the file name without the extension
        file_name_without_extension = os.path.splitext(file_name)[0]
        return file_name_without_extension

# A config that's been executed containing results with images
class Experiment(models.Model):
    config = models.ForeignKey(Config, on_delete=models.CASCADE)
    status = models.CharField(max_length=200, default="created")
    created_at = models.DateTimeField(default=timezone.now)
    def __str__(self):
        return self.created_at.strftime('%Y-%m-%d %H:%M:%S')


class ExplanationResult(models.Model):
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    intput_image = models.ForeignKey(InImage, null=True, blank=True, on_delete=models.CASCADE)
    methods = models.ManyToManyField(ExplanationMethod)
    neural_network = models.CharField(max_length=50)
    date = models.DateField()
    pred_top1 = models.CharField(null=True, blank=True,max_length=50)

# Result for one image
class Result(models.Model):
    explanation_results = models.ForeignKey(ExplanationResult, null=True, blank=True, on_delete=models.CASCADE)
    intput_image = models.ForeignKey(InImage, null=True, blank=True, on_delete=models.CASCADE)
    elapsed_time = models.FloatField(null=True, blank=True, default=0)
    second_pass_pred = models.CharField(null=True, blank=True,max_length=50)
    result_intersect = models.JSONField(null=True, blank=True)
    use_coco = models.BooleanField(null=False, blank=False, default=False)
    coco_categories = models.ManyToManyField(CocoCategories)
    method = models.ForeignKey(ExplanationMethod, null=True, blank=True, on_delete=models.CASCADE)
    final = models.ImageField(null=True, blank=False, upload_to="output_images/final")
    mask = models.ImageField(null=True, blank=False, upload_to="output_images/mask")
    filtered = models.ImageField(null=True, blank=False, upload_to="output_images/filtered")
    coco_masks = models.ImageField(null=True, blank=False, upload_to="output_images/coco_masks")

# Output image
class OutImage(models.Model):
    method = models.ForeignKey(ExplanationMethod, null=True, blank=True, on_delete=models.CASCADE)
    result = models.ForeignKey(Result, null=True, blank=True,  on_delete=models.CASCADE)
    final = models.ImageField(null=True, blank=False, upload_to="output_images/final")
    mask = models.ImageField(null=True, blank=False, upload_to="output_images/mask")
    filtered = models.ImageField(null=True, blank=False, upload_to="output_images/filtered")
    coco_masks = models.ImageField(null=True, blank=False, upload_to="output_images/coco_masks")