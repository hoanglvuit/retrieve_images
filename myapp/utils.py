from pycocotools.coco import COCO
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import pickle
import faiss
from typing import Dict, List, Tuple, Optional
import torch
class ImageSearcher:
    def __init__(
        self, 
        ann_file_path: str,
        model_name: str = 'clip-ViT-B-32',  # Default to CLIP
        embeddings_cache_path: Optional[str] = None,
        use_gpu: bool = False
    ):

        self.coco = COCO(ann_file_path)
        self.model_name = model_name
        
        # If no cache path specified, create one based on model name
        if embeddings_cache_path is None:
            embeddings_cache_path = f'caption_embeddings_{model_name.replace("/", "_")}.pkl'
        self.embeddings_cache_path = embeddings_cache_path
        
        # Initialize the model
        self.model = SentenceTransformer(model_name)
        if use_gpu and torch.cuda.is_available():
            self.model = self.model.to('cuda')
        
        # Load or compute caption embeddings
        self.captions, self.caption_embeddings = self._load_or_compute_embeddings()
        
        # Build FAISS index with appropriate metric based on model
        self.index = self._build_faiss_index()
    
    def _load_or_compute_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Load embeddings from cache if available, otherwise compute and save them"""
        if os.path.exists(self.embeddings_cache_path):
            print("Loading cached embeddings...")
            with open(self.embeddings_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                return cache_data['captions'], cache_data['embeddings']
        
        print("Computing embeddings (this may take a while)...")
        all_ann_ids = self.coco.getAnnIds()
        all_anns = self.coco.loadAnns(all_ann_ids)
        
        captions = [ann['caption'] for ann in all_anns]
        
        # Use batched encoding for better performance
        caption_embeddings = self.model.encode(
            captions,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True  # Important for cosine similarity
        )
        
        print("Saving embeddings to cache...")
        cache_data = {
            'captions': captions,
            'embeddings': caption_embeddings
        }
        with open(self.embeddings_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        return captions, caption_embeddings
    
    def _build_faiss_index(self) -> faiss.Index:
        """Build appropriate FAISS index based on model type"""
        dimension = self.caption_embeddings.shape[1]
        
        # For CLIP and other cosine similarity models
        if self.model_name.startswith('clip-'):
            index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
        else:
            # For other models that might work better with L2 distance
            index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to index
        index.add(self.caption_embeddings)
        return index
    
    def search_images(
        self, 
        query: str, 
        num_images: int = 24,
        threshold: float = None
    ) -> List[Dict]:

        # Encode query with same normalization as captions
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search with larger k to account for duplicates
        search_k = min(num_images * 5, len(self.captions))
        distances, indices = self.index.search(
            query_embedding[np.newaxis, :],
            search_k
        )
        
        # Process results
        all_results = []
        seen_image_ids = set()
        all_ann_ids = self.coco.getAnnIds()
        all_anns = self.coco.loadAnns(all_ann_ids)
        
        for i, idx in enumerate(indices[0]):
            similarity = float(distances[0][i])
            if threshold and similarity < threshold:
                continue
                
            ann = all_anns[idx]
            img_id = ann['image_id']
            
            if img_id not in seen_image_ids:
                img = self.coco.loadImgs(img_id)[0]
                all_results.append({
                    'url': img['coco_url'],
                    'caption': self.captions[idx],
                    'similarity': similarity,
                    'image_id': img_id
                })
                seen_image_ids.add(img_id)
                
                if len(all_results) >= num_images:
                    break
        
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        return all_results[:num_images]