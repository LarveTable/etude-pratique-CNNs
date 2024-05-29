from django.contrib import admin
from .models import Config, InImage, OutImage, Experiment, Result, ExplanationMethod, ExplanationResult, CocoCategories
# Register your database models here.

admin.site.register(Config)
admin.site.register(InImage)
admin.site.register(Experiment)
admin.site.register(OutImage)
admin.site.register(Result)
admin.site.register(ExplanationResult)
admin.site.register(ExplanationMethod)
admin.site.register(CocoCategories)