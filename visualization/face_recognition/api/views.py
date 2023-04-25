from django.shortcuts import render
from rest_framework import viewsets
import numpy as np
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.decorators import action
from api.models import Company, Person, Employee, TestImage, TrainImage
from api.serializers import CompanySerializer, PersonSerializer, EmployeeSerializer, TestImageSerializer, TrainImageSerializer
# Create your views here.
class CompanyViewSet(viewsets.ModelViewSet):
    queryset = Company.objects.all()
    serializer_class=CompanySerializer

    @action(detail=True, methods=['get'])
    def employees(self, request, pk=None):
        print("get Employees of ",pk," company")
        try:
            company = Company.objects.get(pk=pk)
            emps = Employee.objects.filter(company=company)
            emps_serializer = EmployeeSerializer(emps, many=True, context={'request':request})
            return Response(emps_serializer.data)
        except Exception as e:
            print(e)
            return Response({
                'message': "Company or employee dowsnot exist"
            })


class EmployeeViewSet(viewsets.ModelViewSet):
    queryset = Employee.objects.all()
    serializer_class=EmployeeSerializer

class PersonViewSet(viewsets.ModelViewSet):
    queryset = Person.objects.all()
    serializer_class = PersonSerializer

    @action(detail=True, methods=['get'])
    def test_image(self, request, pk=None):
        print("get Employees of ",pk," company")
        try:
            person = Person.objects.get(pk=pk)
            tests = TestImage.objects.filter(person=person)
            tests_serializer = TestImageSerializer(tests, many=True, context={'request':request})
            return Response(tests_serializer.data)
        except Exception as e:
            print(e)
            return Response({
                'message': "Person does not exist"
            })
    @action(detail=True, methods=['get'])
    def train_image(self, request, pk=None):
        print("get Employees of ",pk," company")
        try:
            person = Person.objects.get(pk=pk)
            trains = TrainImage.objects.filter(person=person)
            trains_serializer = TestImageSerializer(trains, many=True, context={'request':request})
            return Response(trains_serializer.data)
        except Exception as e:
            print(e)
            return Response({
                'message': "Person does not exist"
            })

class TestImagenViewSet(viewsets.ModelViewSet):
    queryset = TestImage.objects.all()
    serializer_class = TestImageSerializer

    parser_classes = [MultiPartParser]  # Use MultiPartParser for file upload

    def create(self, request, *args, **kwargs):
        # Get the uploaded file from the request
        image_file = request.FILES.get('image')

        # Call the process_image function to get the matrix representation
        matrix = np.array([[1,2,3],[4,5,6]])

        # Convert the numpy array to a string representation
        matrix_str = np.array2string(matrix, separator=',')

        # Create a TestImagen object with the matrix data
        test_imagen = TestImage(matrix_data=matrix_str)
        test_imagen.save()

        # Serialize the TestImagen object and return the response
        serializer = self.get_serializer(test_imagen)
        return Response(serializer.data)

class TrainImagenViewSet(viewsets.ModelViewSet):
    queryset = TrainImage.objects.all()
    serializer_class = TrainImageSerializer

