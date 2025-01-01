# # image_search/forms.py
from django import forms

class SearchFormfortext(forms.Form):
    query = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your search query...'
        })
    )
    num_images = forms.IntegerField(
        min_value=1,
        max_value=200,
        initial=24,
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
        })
    )

class SearchFormforimage(forms.Form):
    image = forms.ImageField(
        required=True,  # Make this field mandatory
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control',
        })
    )
    num_images = forms.IntegerField(
        min_value=1,
        max_value=200,
        initial=24,
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
        })
    )