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
    active = models.BooleanField(default=True)

    def  __str__(self):
        return self.name+", " + self.location


class Employee(models.Model):
    name = models.CharField(max_length=100)
    email = models.CharField(max_length=50)
    address = models.CharField(max_length=200)
    phone = models.CharField(max_length=10)
    about = models.TextField()
    position = models.CharField(max_length=50, choices=(
        ('Manager', 'manages'), ('Software Developer', 'sd'), ('Project Leader', 'pl')))
    company = models.ForeignKey(Company, on_delete=models.CASCADE)    


class Person(models.Model):
    person_id = models.AutoField(primary_key=True)
    person_name = models.CharField(max_length=50)
    def  __str__(self):
        return self.person_name

class TestImage(models.Model):
    test_image_id = models.AutoField(primary_key=True)
    image = models.ImageField()
    matrix_data = models.TextField(default="[]")
    person = models.ForeignKey(Person, on_delete=models.CASCADE)

class TrainImage(models.Model):
    test_image_id = models.AutoField(primary_key=True)
    image = models.ImageField()
    matrix_data = models.TextField(default="[]")
    person = models.ForeignKey(Person, on_delete=models.CASCADE)