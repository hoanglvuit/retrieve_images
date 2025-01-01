# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('',views.home,name='home'),
    path('searchbyimage/', views.search_byimage, name='search_byimage'),
    path('dataset/',views.dataset,name='dataset'),
    path('searchbytext/',views.search_bytext,name = 'search_bytext')
]
