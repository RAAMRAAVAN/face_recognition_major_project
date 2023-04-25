from rest_framework import serializers
from api.models import Company, Employee, Person, TestImage, TrainImage
# serializers

class CompanySerializer(serializers.HyperlinkedModelSerializer):
    company_id = serializers.ReadOnlyField()
    class Meta:
        model = Company
        fields = "__all__"

class EmployeeSerializer(serializers.HyperlinkedModelSerializer):
    id = serializers.ReadOnlyField()
    class Meta:
        model = Employee
        fields = "__all__"

class PersonSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Person
        fields = "__all__"

class TestImageSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = TestImage
        fields = "__all__"

class TrainImageSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = TrainImage
        fields = "__all__"