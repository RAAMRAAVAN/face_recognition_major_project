"""userproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from home import views
urlpatterns = [
    path('', views.index, name="home"),
    path('homepage', views.homepage, name="homepage"),
    path('accounts', views.accounts, name="accounts"),
    path('vice_chancellor', views.vice_chancellor, name="vice_chancellor"),
    path('college', views.college, name="college"),
    path('report', views.report, name="report"),
    path('campus_planning', views.campus_planning, name="campus_planning"),
    path('login', views.loginUser, name="login"),
    path('logout', views.logoutUser, name="logout"),
]
