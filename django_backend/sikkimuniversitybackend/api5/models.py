from django.db import models

# Create your models here.

# creating company model


class Company(models.Model):
    company_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=50)
    location = models.CharField(max_length=50)
    about = models.TextField()
    type = models.CharField(
        max_length=100, choices=(('IT', 'IT'), ('IT2', 'IT2')))
    added_date = models.DateTimeField(auto_now=True)
    image = models.ImageField()
    active=models.BooleanField(default=True)

class ImageSlider(models.Model):
    ImageSlider_id = models.AutoField(primary_key=True)
    image = models.ImageField()
    about = models.TextField()
    added_date = models.DateTimeField(auto_now=True)

class QuickLinks(models.Model):
    QuickLinks_id = models.AutoField(primary_key=True)
    file = models.FileField()
    text = models.TextField()
    added_date = models.DateTimeField(auto_now=True)

class Objectives_of_Sikkim_University(models.Model):
    Objectives_of_Sikkim_University_id = models.AutoField(primary_key=True)
    objective = models.TextField()
    added_date = models.DateTimeField(auto_now=True)