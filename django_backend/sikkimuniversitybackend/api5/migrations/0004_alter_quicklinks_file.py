# Generated by Django 4.1.7 on 2023-02-23 15:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api5', '0003_quicklinks'),
    ]

    operations = [
        migrations.AlterField(
            model_name='quicklinks',
            name='file',
            field=models.FileField(upload_to=''),
        ),
    ]
