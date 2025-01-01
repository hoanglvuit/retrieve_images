from django.shortcuts import render
from .forms import SearchForm
from .utils import ImageSearcher
from .models import SearchQuery
from django.core.paginator import Paginator
import json
from django.core.serializers.json import DjangoJSONEncoder
searcher = ImageSearcher(r'COCO_DATASET/coco2017/annotations/captions_train2017.json')
