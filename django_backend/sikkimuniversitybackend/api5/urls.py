from django.contrib import admin
from django.urls import path, include
from api5.views import CompanyViewSet, ImageSliderViewSet, QuickLinksViewSet, Objectives_of_Sikkim_UniversityViewSet
from rest_framework import routers

router= routers.DefaultRouter()
router.register(r'companies',CompanyViewSet)
router.register(r'imageslider',ImageSliderViewSet)
router.register(r'quicklinks',QuickLinksViewSet)
router.register(r'Objectives_of_Sikkim_University',Objectives_of_Sikkim_UniversityViewSet)

urlpatterns = [
    path('',include(router.urls))
]
