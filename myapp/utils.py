from pycocotools.coco import COCO
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
import nltk
import spacy
import numpy as np
import os
import pickle
import faiss
class ImageSearcher:
    def __init__(self, ann_file_path, embeddings_cache_path='caption_embeddings_faiss.pkl'):
        self._download_nltk_data()
        
        self.coco = COCO(ann_file_path)
        self.model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        self.stop_words = set(stopwords.words('english'))
        self.embeddings_cache_path = embeddings_cache_path
        
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_md')
        except:
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_md'])
            self.nlp = spacy.load('en_core_web_md')
        
        # Cache for word similarities
        self.similarity_cache = {}
        
        # Load or compute caption embeddings
        self.captions, self.caption_embeddings = self._load_or_compute_embeddings()
        
        # Build FAISS index
        self.index = self._build_faiss_index()

    def _download_nltk_data(self):
        """Download all required NLTK data"""
        required_nltk_data = [
            'punkt',
            'stopwords',
            'wordnet',
            'averaged_perceptron_tagger',
            'omw-1.4'
        ]
        
        for item in required_nltk_data:
            try:
                nltk.data.find(f'tokenizers/{item}' if item == 'punkt' else f'corpora/{item}')
            except LookupError:
                print(f"Downloading {item}...")
                nltk.download(item, quiet=True)

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
        
        # Extract and process captions
        captions = [ann['caption'] for ann in all_anns]
        
        # Process captions with lemmatization
        processed_captions = [self._lemmatize_text(caption) for caption in captions]
        
        # Compute embeddings for processed captions
        caption_embeddings = self.model.encode(
            processed_captions, 
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
        # Assuming self.caption_embeddings is a PyTorch tensor
        caption_embeddings_cpu = self.caption_embeddings  # Move to CPU and convert to NumPy
        index = faiss.IndexFlatL2(caption_embeddings_cpu.shape[1])  # Create a FAISS index
        index.add(caption_embeddings_cpu)  # Add all embeddings to the index
        return index


    def _lemmatize_text(self, text):
        """Process text with spaCy for lemmatization and cleaning"""
        doc = self.nlp(text.lower())
        lemmatized_tokens = [
            token.lemma_ 
            for token in doc 
            if (not token.is_stop and 
                not token.is_punct and 
                not token.is_space and
                token.lemma_.strip())
        ]
        return ' '.join(lemmatized_tokens)
    
    def process_query(self, query):
        """Process query with lemmatization and basic cleaning"""
        processed_query = self._lemmatize_text(query)
        if not processed_query:
            return [query]
        return [processed_query]

    def search_images(self, query, num_images=24):
        processed_queries = self.process_query(query)
        all_results = []
        seen_image_ids = set()
        
        all_ann_ids = self.coco.getAnnIds()
        all_anns = self.coco.loadAnns(all_ann_ids)
        
        for processed_query in processed_queries:
            query_embedding = self.model.encode(processed_query, convert_to_numpy=True)
            
            # Increase the search range to ensure we find enough unique images
            search_range = min(num_images * 5, len(all_anns))
            
            # Use FAISS to find the nearest neighbors
            distances, indices = self.index.search(query_embedding[np.newaxis, :], search_range)
            
            for i, idx in enumerate(indices[0]):
                ann = all_anns[idx]
                img_id = ann['image_id']
                
                if img_id not in seen_image_ids:
                    img = self.coco.loadImgs(img_id)[0]
                    all_results.append({
                        'url': img['coco_url'],
                        'caption': self.captions[idx],
                        'similarity': float(-distances[0][i]),  # Convert distance to similarity
                        'matched_query': processed_query
                    })
                    seen_image_ids.add(img_id)
                    
                    # Stop when we've found the desired number of unique images
                    if len(all_results) >= num_images:
                        break
            
            # Stop searching if we've found enough images
            if len(all_results) >= num_images:
                break
        
        # Ensure we return exactly the number of images requested
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        return all_results[:num_images]