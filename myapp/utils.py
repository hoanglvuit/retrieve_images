from pycocotools.coco import COCO
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import pickle
import faiss

class ImageSearcher:
    def __init__(self, ann_file_path, embeddings_cache_path='caption_embeddings_faiss_3.pkl'):
        self.coco = COCO(ann_file_path)
        self.model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        self.embeddings_cache_path = embeddings_cache_path
        
        # Load or compute caption embeddings
        self.captions, self.caption_embeddings = self._load_or_compute_embeddings()
        
        # Build FAISS index
        self.index = self._build_faiss_index()

    def _load_or_compute_embeddings(self):
        """Load embeddings from cache if available, otherwise compute and save them"""
        if os.path.exists(self.embeddings_cache_path):
            print("Loading cached embeddings...")
            with open(self.embeddings_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                return cache_data['captions'], cache_data['embeddings']
        
        print("Computing embeddings (this may take a while)...")
        # Get all annotations
        all_ann_ids = self.coco.getAnnIds()
        all_anns = self.coco.loadAnns(all_ann_ids)
        
        # Extract captions
        captions = [ann['caption'] for ann in all_anns]
        
        # Compute embeddings for captions
        caption_embeddings = self.model.encode(
            captions, 
            convert_to_numpy=True, 
            show_progress_bar=True
        )
        
        # Save to cache
        print("Saving embeddings to cache...")
        cache_data = {
            'captions': captions,  # Store original captions for display
            'embeddings': caption_embeddings
        }
        with open(self.embeddings_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        return captions, caption_embeddings

    def _build_faiss_index(self):
        # Assuming self.caption_embeddings is a NumPy array
        index = faiss.IndexFlatL2(self.caption_embeddings.shape[1])  # Create a FAISS index
        index.add(self.caption_embeddings)  # Add all embeddings to the index
        return index
    def search_images(self, query, num_images=24):
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Increase the search range to ensure we find enough unique images
        search_range = min(num_images * 5, len(self.captions))
        
        # Use FAISS to find the nearest neighbors
        distances, indices = self.index.search(query_embedding[np.newaxis, :], search_range)
        
        all_results = []
        seen_image_ids = set()
        all_ann_ids = self.coco.getAnnIds()
        all_anns = self.coco.loadAnns(all_ann_ids)

        for i, idx in enumerate(indices[0]):
            ann = all_anns[idx]
            img_id = ann['image_id']
            
            if img_id not in seen_image_ids:
                img = self.coco.loadImgs(img_id)[0]
                all_results.append({
                    'url': img['coco_url'],
                    'caption': self.captions[idx],
                    'similarity': float(-distances[0][i])  # Convert distance to similarity
                })
                seen_image_ids.add(img_id)
                
                # Stop when we've found the desired number of unique images
                if len(all_results) >= num_images:
                    break
        
        # Ensure we return exactly the number of images requested
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        return all_results[:num_images]
