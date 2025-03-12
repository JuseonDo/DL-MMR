from typing import List
import gc
import faiss
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class FaissSearch:
    def __init__(self, 
        train_data:List[str],
        test_data:List[str],
        model_name_or_path: str = 'facebook/bart-large',
        tokenizer_name_or_path: str = 'facebook/bart-large',
        device: str = 'cuda',
        batch_size: int = 128,
        similarity_type:str = "distance", # cos_similarity, distance
        ) -> None:

        self.device = device
        self.batch_size = batch_size
        self.train_data = train_data
        self.test_data = test_data
        self.similarity_type = similarity_type


        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

        self.train_dataset = None
        self.train_embedd = self.get_embeddings(self.train_data)
        self.train_embedd = np.array(self.train_embedd).astype(np.float32)

        if test_data is not None:
            self.test_embedd = self.get_embeddings(self.test_data)
            self.test_embedd = np.array(self.test_embedd).astype(np.float32)

        
        dimension = self.train_embedd.shape[1]
        if self.similarity_type == "cos_similarity":
            self.faiss_index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(self.train_embedd)
            self.faiss_index.add(self.train_embedd)
        else:
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(self.train_embedd)

        if self.similarity_type == "cos_similarity" and self.test_embedd is not None:
            faiss.normalize_L2(self.test_embedd)
    
    def get_embeddings(self,
        data:List[str],
    ) -> torch.Tensor:
        
        gc.collect()
        torch.cuda.empty_cache()
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0,len(data),self.batch_size)):
                batch = data[i : i+self.batch_size]
                encoded_text = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                ).to(self.device)
                self.model = self.model.to(self.device)
                embedding = self.model(**encoded_text).last_hidden_state.mean(dim=1).detach().cpu().tolist()
                
                embeddings.extend(embedding)

        return embeddings
