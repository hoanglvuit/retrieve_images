#views.py
from django.shortcuts import render
from .forms import SearchForm
from .utils import ImageSearcher
from .models import SearchQuery
from django.core.paginator import Paginator
import json
from django.core.serializers.json import DjangoJSONEncoder
# Initialize the ImageSearcher with your COCO annotations file path
searcher = ImageSearcher(r'D:\CS336_IR\IR_Project\myproject\COCO_DATASET\coco2017\annotations\captions_train2017.json')
def home(request):
    return render(request, 'home.html')
def search(request):
    results = []
    original_query = ""
    processed_query = ""
    num_images = 24  # Default number of images
    
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            query = form.cleaned_data['query']
            num_images = form.cleaned_data.get('num_images', 24)
            original_query = query
            processed_query = searcher.process_query(query)
            
            # Save the search query
            SearchQuery.objects.create(query=query)
            
            # Get search results with specified number of images
            results = searcher.search_images(query, num_images=num_images)
    else:
        form = SearchForm()
    
    return render(request, 'search.html', {
        'form': form,
        'results': results,
        'original_query': original_query,
        'processed_query': processed_query,
        'num_images': num_images,
        'total_results': len(results)
    })

import json
from django.core.serializers.json import DjangoJSONEncoder

def dataset(request):
    page_number = request.GET.get('page', 1)
    items_per_page = 35
    
    searcher = ImageSearcher(r'D:\CS336_IR\IR_Project\myproject\COCO_DATASET\coco2017\annotations\captions_train2017.json')
    
    all_img_ids = searcher.coco.getImgIds()
    dataset_items = []
    
    for img_id in all_img_ids:
        img = searcher.coco.loadImgs(img_id)[0]
        ann_ids = searcher.coco.getAnnIds(imgIds=img_id)
        anns = searcher.coco.loadAnns(ann_ids)
        
        # Properly serialize captions for JavaScript
        captions = [ann['caption'] for ann in anns]
        captions_json = json.dumps(captions, cls=DjangoJSONEncoder)
        
        dataset_items.append({
            'image_id': img_id,
            'url': img['coco_url'],
            'captions': captions_json,  # Now properly serialized
            'width': img['width'],
            'height': img['height']
        })
    
    paginator = Paginator(dataset_items, items_per_page)
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'total_images': len(all_img_ids),
        'current_page': int(page_number),
        'total_pages': paginator.num_pages,
    }
    
    return render(request, 'dataset.html', context)