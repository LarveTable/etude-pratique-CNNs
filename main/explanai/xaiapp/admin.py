from django.contrib import admin
from .models import Config, Image
# Register your database models here.

admin.site.register(Config)
admin.site.register(Image)