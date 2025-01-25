from pycocotools.coco import COCO
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import pickle
import faiss
from typing import Dict, List, Tuple, Optional
import torch
from PIL import Image
class ImageSearcher_COCO:
    def __init__(
        self, 
        ann_file_path: str,
        model_name: str = 'clip-ViT-B-32',  
        embeddings_cache_path: Optional[str] = None,
        use_gpu: bool = False
    ):
        self.coco = COCO(ann_file_path)
        self.model_name = model_name
        if embeddings_cache_path is None:
            embeddings_cache_path = f'caption_embeddings_{model_name.replace("/", "_")}.pkl'
        self.embeddings_cache_path = embeddings_cache_path
        self.model = SentenceTransformer(model_name)
        if use_gpu and torch.cuda.is_available():
            self.model = self.model.to('cuda')
        print( self.embeddings_cache_path)
        self.captions, self.caption_embeddings = self._load_or_compute_embeddings()
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
        image_id = [ann['image_id'] for ann in all_anns]

        caption_embeddings = self.model.encode(
            captions,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True  
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
        if self.model_name.startswith('clip-'):
            index = faiss.IndexFlatIP(dimension)  
        else:
            index = faiss.IndexFlatL2(dimension)
        index.add(self.caption_embeddings)
        return index
    def search_images(
        self,
        #query: str, 
        query,
        num_images: int = 24,
        threshold: float = None
    ) -> List[Dict]:
        # image = 'images.jpg'
        # query_embedding = self.model.encode(
        #     Image.open(image),
        #     convert_to_numpy=True,
        #     normalize_embeddings=True
        # )
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        search_k = min(num_images * 5, len(self.captions))
        distances, indices = self.index.search(
            query_embedding[np.newaxis, :],
            search_k
        )
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


class ImageSearcher_humanface:
    def __init__(
            self,
            image_folder: str,
            model_name: str = 'clip-ViT-B-32',
            embeddings_cache_path: Optional[str] = None,
            use_gpu: bool = False
    ):
        self.image_folder = image_folder
        self.model_name = model_name
        self.embeddings_cache_path = embeddings_cache_path or f'image_embeddings_{model_name.replace("/", "_")}.pkl'
        self.model = SentenceTransformer(model_name)
        if use_gpu and torch.cuda.is_available():
            self.model = self.model.to('cuda')
        self.image_paths, self.image_embeddings = self._load_or_compute_embeddings()
        self.index = self._build_faiss_index()

    def _load_or_compute_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Load embeddings from cache if available, otherwise compute and save them"""
        if os.path.exists(self.embeddings_cache_path):
            print("Loading cached embeddings...")
            with open(self.embeddings_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                return cache_data['image_paths'], cache_data['embeddings']

        print("Computing embeddings (this may take a while)...")
        image_paths = [os.path.join(self.image_folder, fname) for fname in os.listdir(self.image_folder) if
                       fname.endswith(('.jpg', '.png', '.jpeg'))]
        image_embeddings = []

        for img_path in image_paths:
            image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
            embedding = self.model.encode(
                image,  # Pass the image directly to the model
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            image_embeddings.append(embedding)

        image_embeddings = np.array(image_embeddings)
        print("Saving embeddings to cache...")
        cache_data = {
            'image_paths': image_paths,
            'embeddings': image_embeddings
        }
        with open(self.embeddings_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        return image_paths, image_embeddings

    def _build_faiss_index(self) -> faiss.Index:
        """Build FAISS index for efficient similarity search"""
        dimension = self.image_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(self.image_embeddings)
        return index

    def search(self, query, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar images using a query image"""
        #query_image = Image.open(query_image_path).convert('RGB')
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        query_embedding = np.array([query_embedding])  # Reshape for FAISS
        distances, indices = self.index.search(query_embedding, top_k)
        return [(self.image_paths[idx], float(dist)) for idx, dist in zip(indices[0], distances[0])]

