#views.py
from django.shortcuts import render
from .forms import SearchFormfortext,SearchFormforimage
from .utils import ImageSearcher
from .models import SearchQuery, UploadedImage
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
    uploaded_image_url = None
    results = []
    total_results = 0
    time_taken = 0

    if request.method == 'POST':
        form = SearchFormforimage(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Save the uploaded image
                uploaded_image = form.cleaned_data['image']
                image_model = UploadedImage(image=uploaded_image)
                image_model.save()

                # Get the URL for displaying the uploaded image
                uploaded_image_url = image_model.image.url

                # Get number of images to show
                num_images = form.cleaned_data.get('num_images', 24)

                # Start timing
                start_time = time.time()

                # Open the uploaded image with PIL
                query_image = Image.open(image_model.image.path)

                # Use the existing searcher instance to perform the search
                results = searcher.search_images(
                    query=query_image,
                    num_images=num_images
                )

                # Calculate time taken
                time_taken = round(time.time() - start_time, 2)
                total_results = len(results)

            except Exception as e:
                print(f"Error during image search: {str(e)}")
                # You might want to add error handling here or show an error message to the user
        else:
            print("Form is invalid:", form.errors)
    else:
        form = SearchFormforimage()

    return render(
        request,
        'searchbyimage.html',
        {
            'form': form,
            'uploaded_image_url': uploaded_image_url,
            'results': results,
            'total_results': total_results,
            'time': time_taken,
        },
    )
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