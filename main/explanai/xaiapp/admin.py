from django.contrib import admin
from .models import Config, Image, OutImage, Experiment, Result, Stat
# Register your database models here.

admin.site.register(Config)
admin.site.register(Image)
admin.site.register(Experiment)
admin.site.register(OutImage)
admin.site.register(Result)
admin.site.register(Stat)