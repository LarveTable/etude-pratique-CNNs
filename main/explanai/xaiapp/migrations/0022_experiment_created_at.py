# Generated by Django 5.0.6 on 2024-06-03 22:31

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('xaiapp', '0021_merge_20240603_2153'),
    ]

    operations = [
        migrations.AddField(
            model_name='experiment',
            name='created_at',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
    ]