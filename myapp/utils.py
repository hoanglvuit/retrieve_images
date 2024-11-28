from pycocotools.coco import COCO
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
import nltk
import spacy
import numpy as np
import os
import pickle

class ImageSearcher:
    def __init__(self, ann_file_path, embeddings_cache_path='caption_embeddings_test.pkl'):
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
            convert_to_tensor=True, 
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
    
    def _lemmatize_text(self, text):
        """Process text with spaCy for lemmatization and cleaning"""
        # Process the text with spaCy
        doc = self.nlp(text.lower())
        
        # Get lemmatized tokens, excluding stopwords and punctuation
        lemmatized_tokens = [
            token.lemma_ 
            for token in doc 
            if (not token.is_stop and 
                not token.is_punct and 
                not token.is_space and
                token.lemma_.strip())
        ]
        
        # Join tokens back into a string
        return ' '.join(lemmatized_tokens)
    
    def process_query(self, query):
        """Process query with lemmatization and basic cleaning"""
        # Lemmatize the query
        processed_query = self._lemmatize_text(query)
        
        # If the processed query is empty, return the original query
        if not processed_query:
            return [query]
        
        return [processed_query]  # Return as list for compatibility
    
    def search_images(self, query, num_images=24, batch_size=512):
        processed_queries = self.process_query(query)
        all_results = []
        seen_image_ids = set()
        
        # Load all annotations once
        all_ann_ids = self.coco.getAnnIds()
        all_anns = self.coco.loadAnns(all_ann_ids)
        
        for processed_query in processed_queries:
            query_embedding = self.model.encode(processed_query, convert_to_tensor=True)
            
            # Calculate similarities with cached embeddings
            similarities = util.pytorch_cos_sim(query_embedding, self.caption_embeddings).squeeze().cpu().numpy()
            
            # Get top K similar captions
            sorted_indices = np.argsort(similarities)[-num_images:][::-1]
            
            for idx in sorted_indices:
                ann = all_anns[idx]
                img_id = ann['image_id']
                
                if img_id not in seen_image_ids:
                    img = self.coco.loadImgs(img_id)[0]
                    all_results.append({
                        'url': img['coco_url'],
                        'caption': self.captions[idx],  # Use original caption for display
                        'similarity': float(similarities[idx]),
                        'matched_query': processed_query
                    })
                    seen_image_ids.add(img_id)
        
        # Sort all results by similarity score
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        return all_results[:num_images]