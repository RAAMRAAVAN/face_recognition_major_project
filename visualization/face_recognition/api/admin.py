from django.contrib import admin
from api.models import Company, Person, Employee, TestImage, TrainImage
# Register your models here.
admin.site.register(Company)
admin.site.register(Employee)
admin.site.register(Person)
admin.site.register(TestImage)
admin.site.register(TrainImage)