from django import forms

class ConfigForm(forms.Form):
    modelName = forms.CharField()
    image = forms.ImageField()