from django.shortcuts import render
from rest_framework import viewsets
from api5.models import Company, ImageSlider, QuickLinks, Objectives_of_Sikkim_University
from api5.serializers import CompanySerializer, ImageSliderSerializer, QuickLinksSerializer, Objectives_of_Sikkim_University_Serializer
# Create your views here.
class CompanyViewSet(viewsets.ModelViewSet):
    queryset = Company.objects.all()
    serializer_class=CompanySerializer

class ImageSliderViewSet(viewsets.ModelViewSet):
    queryset = ImageSlider.objects.all()
    serializer_class=ImageSliderSerializer

class QuickLinksViewSet(viewsets.ModelViewSet):
    queryset = QuickLinks.objects.all()
    serializer_class=QuickLinksSerializer

class Objectives_of_Sikkim_UniversityViewSet(viewsets.ModelViewSet):
    queryset = Objectives_of_Sikkim_University.objects.all()
    serializer_class=Objectives_of_Sikkim_University_Serializer