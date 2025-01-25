from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('coco/', views.coco, name='coco'),
    path('human_face/', views.human_face, name='human_face'),
    path('coco/searchbyimage/', views.search_byimage, name='search_byimage'),
    path('coco/dataset/', views.dataset, name='dataset'),
    path('coco/searchbytext/', views.search_bytext, name='search_bytext'),
    path('human_face/searchbyimage/', views.search_byimage1, name='search_byimage1'),
    path('human_face/dataset/', views.dataset, name='dataset'),
    path('human_face/searchbytext/', views.search_bytext1, name='search_bytext1')
]

# Serve static files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)