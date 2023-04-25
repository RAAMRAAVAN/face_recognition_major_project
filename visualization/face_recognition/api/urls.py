from django.contrib import admin
from django.urls import path, include
from api.views import CompanyViewSet, PersonViewSet, EmployeeViewSet, TestImagenViewSet, TrainImagenViewSet
from rest_framework import routers

router= routers.DefaultRouter()
router.register(r'companies',CompanyViewSet)
router.register(r'employees',EmployeeViewSet)
router.register(r'person', PersonViewSet)
router.register(r'test_image', TestImagenViewSet)
router.register(r'train_image', TrainImagenViewSet)

urlpatterns = [
    path('',include(router.urls))
]
