# Generated by Django 5.0.4 on 2024-05-17 20:49

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('xaiapp', '0005_alter_image_image'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='outimage',
            name='result',
        ),
        migrations.CreateModel(
            name='InImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('status', models.CharField(default='pending', max_length=200)),
                ('image', models.ImageField(upload_to='input_images/')),
                ('config', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='xaiapp.config')),
            ],
        ),
        migrations.AddField(
            model_name='outimage',
            name='input_image',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, to='xaiapp.inimage'),
            preserve_default=False,
        ),
        migrations.DeleteModel(
            name='Image',
        ),
    ]