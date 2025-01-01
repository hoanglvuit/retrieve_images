#views.py
from django.shortcuts import render
from .forms import SearchFormfortext,SearchFormforimage
from .utils import ImageSearcher
from .models import SearchQuery
from django.core.paginator import Paginator
import json
from PIL import Image
import time
from django.core.serializers.json import DjangoJSONEncoder
# Initialize the ImageSearcher with your COCO annotations file path
searcher = ImageSearcher(r'COCO_DATASET/coco2017/annotations/captions_train2017.json')
def home(request):
    return render(request, 'home.html')
def search_byimage(request):
    results = []
    num_images = 24  # Default number of images
    times = 0
    if request.method == 'POST':
        form = SearchFormforimage(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.cleaned_data['image']
            
            # Process the uploaded image (e.g., extract features or descriptors)
            image = Image.open(uploaded_image)
            num_images = form.cleaned_data.get('num_images', 24)
            # Example: Pass the image to a searcher for visual-based search
            # Assuming `search_images` can handle image input
            start = time.time()
            results = searcher.search_images(image, num_images=num_images)
            end =  time.time()
            times = end - start
            # Save the search action if needed (optional, for logging)
            SearchQuery.objects.create(query="Image Search")

    else:
        form = SearchFormforimage()
    
    return render(request, 'searchbyimage.html', {
        'form': form,
        'results': results,
        'num_images': num_images,
        'total_results': len(results),
        'time' : times,
    })


def search_bytext(request):
    results = []
    original_query = ""
    num_images = 24  # Default number of images
    times = 0
    if request.method == 'POST':
        form = SearchFormfortext(request.POST)
        if form.is_valid():
            query = form.cleaned_data['query']
            num_images = form.cleaned_data.get('num_images', 24)
            original_query = query

            # Save the search query
            SearchQuery.objects.create(query=query)
            start = time.time()
            # Get search results with specified number of images
            results = searcher.search_images(query, num_images=num_images)
            end = time.time()
            times = end - start
    else:
        form = SearchFormfortext()

    return render(request, 'searchbytext.html', {
        'form': form,
        'results': results,
        'original_query': original_query,
        'num_images': num_images,
        'total_results': len(results),
        'time' : times,
    })

import json
from django.core.serializers.json import DjangoJSONEncoder

def dataset(request):
    page_number = request.GET.get('page', 1)
    items_per_page = 35
    
    searcher = ImageSearcher(r'COCO_DATASET/coco2017/annotations/captions_train2017.json')
    
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