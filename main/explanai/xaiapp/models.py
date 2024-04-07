from django.db import models

class Config(models.Model):
    model_name = models.CharField(max_length=50)

    def __str__(self) -> str:
        return super().__str__() + self.model_name


class Image(models.Model):
    config = models.ForeignKey(Config, on_delete=models.CASCADE)
    image = models.ImageField(null=False, blank=False, upload_to="input_images/")