# Generated by Django 5.0.6 on 2024-06-03 23:18

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('xaiapp', '0022_experiment_created_at'),
    ]

    operations = [
        migrations.AddField(
            model_name='explanationresult',
            name='intput_image',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='xaiapp.inimage'),
        ),
    ]