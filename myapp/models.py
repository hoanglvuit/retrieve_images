# models.py
from django.db import models

# Create your models here.
from django.db import models

class SearchQuery(models.Model):
    objects = None
    query = models.CharField(max_length=200)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.query} - {self.timestamp}"