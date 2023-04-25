from rest_framework import serializers
from api5.models import Company
from api5.models import ImageSlider, QuickLinks, Objectives_of_Sikkim_University
# serializers


class CompanySerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Company
        fields = "__all__"

class ImageSliderSerializer(serializers.HyperlinkedModelSerializer):
    ImageSlider_id = serializers.ReadOnlyField()
    class Meta:
        model = ImageSlider
        fields = "__all__"

class QuickLinksSerializer(serializers.HyperlinkedModelSerializer):
    QuickLinks_id = serializers.ReadOnlyField()
    class Meta:
        model = QuickLinks
        fields = "__all__"

class Objectives_of_Sikkim_University_Serializer(serializers.HyperlinkedModelSerializer):
    Objectives_of_Sikkim_University_id = serializers.ReadOnlyField()
    class Meta:
        model = Objectives_of_Sikkim_University
        fields = "__all__"