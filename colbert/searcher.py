import os
import torch
import numpy as np

from tqdm import tqdm
from typing import Union
from scipy.stats import rankdata
from colbert.data import Collection, Queries, Ranking

from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.colbert import colbert_score, colbert_score_rescore, colbert_score_reduce_rescore,colbert_score_positive,colbert_score_negative,improved_colbert_score, colbert_score_original,colbert_score_sequence_aware,colbert_score_coverage,colbert_score_coverage_for_batch
from colbert.search.index_storage import IndexScorer

from colbert.infra.provenance import Provenance
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert.infra.launcher import print_memory_stats

import time

TextQueries = Union[str, 'list[str]', 'dict[int, str]', Queries]


def calculate_query_weight(original_query, rewritten_query):
    original_length = len(original_query.split())
    rewritten_length = len(rewritten_query.split())
    length_ratio = rewritten_length / original_length

    weight_new = 0.4 * (1 / (1 + np.exp(-length_ratio + 1))) + 0.3
    weight_original = 1 - weight_new

    return weight_original, weight_new

class Searcher:
    def __init__(self, index, checkpoint=None, collection=None, config=None, index_root=None, verbose: int = 3):


        self.verbose = verbose
        if self.verbose > 1:
            print_memory_stats()

        initial_config = ColBERTConfig.from_existing(config, Run().config)

        default_index_root = initial_config.index_root_
        index_root = index_root if index_root else default_index_root
        self.index = os.path.join(index_root, index)
        self.index_config = ColBERTConfig.load_from_index(self.index)

        self.checkpoint = checkpoint or self.index_config.checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(self.checkpoint)
        self.config = ColBERTConfig.from_existing(self.checkpoint_config, self.index_config, initial_config)
        print(f"index: {index}")  
        print(f"checkpoint: {checkpoint}")  
        print(f"collection: {collection}") 
        self.collection = Collection.cast(collection or self.config.collection)
        self.configure(checkpoint=self.checkpoint, collection=self.collection)

        self.checkpoint = Checkpoint(self.checkpoint, colbert_config=self.config, verbose=self.verbose)
        use_gpu = self.config.total_visible_gpus > 0
        if use_gpu:
            self.checkpoint = self.checkpoint.cuda()
        load_index_with_mmap = self.config.load_index_with_mmap
        if load_index_with_mmap and use_gpu:
            raise ValueError(f"Memory-mapped index can only be used with CPU!")
        self.ranker = IndexScorer(self.index, use_gpu, load_index_with_mmap)

        print_memory_stats()


    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encode(self, text: TextQueries, full_length_search=False):
        queries = text if type(text) is list else [text]
        bsize = 128 if len(queries) > 128 else None

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        Q = self.checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True, full_length_search=full_length_search)

        return Q

    def search(self, text: str, k=10, filter_fn=None, full_length_search=False, pids=None):
        Q = self.encode(text, full_length_search=full_length_search)
        return self.dense_search(Q, k, filter_fn=filter_fn, pids=pids)

    def search_all(self, queries: TextQueries, k=10, filter_fn=None, full_length_search=False, qid_to_pids=None):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        Q = self.encode(queries_, full_length_search=full_length_search)

        return self._search_all_Q(queries, Q, k, filter_fn=filter_fn, qid_to_pids=qid_to_pids)

    def _search_all_Q(self, queries, Q, k, filter_fn=None, qid_to_pids=None):
        qids = list(queries.keys())

        if qid_to_pids is None:
            qid_to_pids = {qid: None for qid in qids}

        all_scored_pids = [
            list(
                zip(
                    *self.dense_search(
                        Q[query_idx:query_idx+1],
                        k, filter_fn=filter_fn,
                        pids=qid_to_pids[qid]
                    )
                )
            )
            for query_idx, qid in tqdm(enumerate(qids))
        ]

        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}

        provenance = Provenance()
        provenance.source = 'Searcher::search_all'
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data, provenance=provenance)

    def dense_search(self, Q: torch.Tensor, k=10, filter_fn=None, pids=None):
        if k <= 10:
            if self.config.ncells is None:
                self.configure(ncells=1)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.5)
            if self.config.ndocs is None:
                self.configure(ndocs=256)
        elif k <= 100:
            if self.config.ncells is None:
                self.configure(ncells=2)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.45)
            if self.config.ndocs is None:
                self.configure(ndocs=1024)
        else:
            if self.config.ncells is None:
                self.configure(ncells=4)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.4)
            if self.config.ndocs is None:
                self.configure(ndocs=max(k * 4, 4096))

        pids, scores = self.ranker.rank(self.config, Q, filter_fn=filter_fn, pids=pids)

        return pids[:k], list(range(1, k+1)), scores[:k]

    def search_with_blend_rewrite_query(self, condense_query: str, rewrite_text: str, k=10):
            # Concatenate condense query and rewrite query
            combined_query = condense_query + " " + rewrite_text

            # Encode the combined query
            combined_encoded = self.encode(combined_query, full_length_search=False)

            # Ensure condense_query and rewrite_text are lists for tokenization
            condense_query_tokens = self.checkpoint.query_tokenizer.tokenize([condense_query])
            split_point = len(condense_query_tokens[0])

            # Ensure split_point does not exceed the length of combined_encoded
            if split_point > combined_encoded.size(1):
                split_point = combined_encoded.size(1)

            # If split_point is equal to the length, use the entire encoded vector
            if split_point == combined_encoded.size(1):
                rewrite_encoded = combined_encoded
            else:
                # Extract only the rewrite part of the encoded vector
                rewrite_encoded = combined_encoded[:, split_point:]

            # Debugging information
            print(f"Combined Query: {combined_query}")
            print(f"Combined Encoded Shape: {combined_encoded.shape}")
            print(f"Split Point: {split_point}")
            print(f"Rewrite Encoded Shape: {rewrite_encoded.shape}")

            # Perform dense search using only the rewrite_encoded vector
            return self.dense_search(rewrite_encoded, k)
    
    def rerank_topk(self, query_text, topk_pids, rerank_top_k=100):
        return self.minibatch_search_and_rescore(query_text, topk_pids, rerank_top_k=100)
    
    def minibatch_search_and_rescore(self, query_text, topk_pids, rerank_top_k=100):
        Q = self.checkpoint.queryFromText([query_text], bsize=32)[0]
        Q = Q.unsqueeze(0)

        topk_docs = [self.collection[pid] for pid in topk_pids[:rerank_top_k]]
        D = self.checkpoint.docFromText(topk_docs, bsize=32)[0]
        D_mask = torch.ones(D.shape[:2], dtype=int)

        scores = colbert_score(Q, D, D_mask)

        sorted_indices = torch.argsort(scores, descending=True).tolist()  # 降序排序
        sorted_pids = [topk_pids[i] for i in sorted_indices]
        sorted_scores = scores[sorted_indices].tolist()

        return sorted_pids, sorted_scores
    
    def minibatch_search_with_blend_rewrite_query_and_rescore(self, condense_query: str, rewrite_text: str, topk_pids, rerank_top_k=100):
        combined_query = condense_query + " " + rewrite_text
        combined_encoded = self.encode(combined_query, full_length_search=False)

        condense_query_tokens = self.checkpoint.query_tokenizer.tokenize([condense_query])
        split_point = len(condense_query_tokens[0])

        if split_point > combined_encoded.size(1):
            split_point = combined_encoded.size(1)

        if split_point == combined_encoded.size(1):
            rewrite_encoded = combined_encoded
        else:
            rewrite_encoded = combined_encoded[:, split_point:]

        print(f"Combined Query 2: {combined_query}")
        print(f"Split Point 2: {split_point}")
        # print(f"Rewrite Encoded Shape 2: {rewrite_encoded.shape}")

        Q = rewrite_encoded[0].unsqueeze(0)

        topk_docs = [self.collection[pid] for pid in topk_pids[:rerank_top_k]]
        D = self.checkpoint.docFromText(topk_docs, bsize=32)[0]
        D_mask = torch.ones(D.shape[:2], dtype=int)

        scores = colbert_score(Q, D, D_mask)

        sorted_indices = torch.argsort(scores, descending=True).tolist()  # 降序排序
        sorted_pids = [topk_pids[i] for i in sorted_indices]
        sorted_scores = scores[sorted_indices].tolist()

        return sorted_pids, sorted_scores
    
    

    
    def minibatch_search_relative_rescore(self, condense_query: str, original_query: str, topk_pids, original_scores, rerank_top_k=100):
        print(f"Original query: {original_query}")
        print(f"Condensed query: {condense_query}")

        encoded = self.encode(condense_query, full_length_search=False)
        Q = encoded[0].unsqueeze(0)

        topk_docs = [self.collection[pid] for pid in topk_pids[:rerank_top_k]]
        D = self.checkpoint.docFromText(topk_docs, bsize=32)[0]
        D_mask = torch.ones(D.shape[:2], dtype=torch.bool)

        new_scores = colbert_score_rescore(Q, D, D_mask, self.config).cpu().numpy()
        original_scores = np.array(original_scores[:rerank_top_k])

        weight_original, weight_new = calculate_query_weight(original_query, condense_query)
        print(f"Weight for original scores: {weight_original:.4f}")
        print(f"Weight for new scores: {weight_new:.4f}")

        original_ranks = rankdata(-original_scores, method='ordinal')
        new_ranks = rankdata(-new_scores, method='ordinal')

        combined_ranks = weight_original * original_ranks + weight_new * new_ranks

        sorted_indices = np.argsort(combined_ranks)
        sorted_pids = [topk_pids[i] for i in sorted_indices]

        max_score = max(np.max(original_scores), np.max(new_scores))
        min_score = min(np.min(original_scores), np.min(new_scores))
        sorted_scores = np.linspace(max_score, min_score, len(sorted_indices))

        return sorted_pids, sorted_scores.tolist()

    def p_rescore(self, condense_query: str, original_query: str, topk_pids, original_scores, rerank_top_k=100):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        main_encoded = self.encode(original_query, full_length_search=False).to(device)
        doc_maxlen = self.config.doc_maxlen
        D_padded_result = self.checkpoint.docFromText([self.collection[pid] for pid in topk_pids], bsize=32, keep_dims=True)

        D_padded = D_padded_result[0] if isinstance(D_padded_result, tuple) else D_padded_result

        D_mask = torch.ones(D_padded.shape[:-1], dtype=torch.int, device=D_padded.device)
        
        
        positive_scores = colbert_score_sequence_aware(main_encoded, D_padded, D_mask)


        sorted_indices = torch.argsort(positive_scores, descending=True).tolist()  # 降序排序
        sorted_pids = [topk_pids[i] for i in sorted_indices]
        sorted_scores = positive_scores[sorted_indices].tolist()


        return sorted_pids, sorted_scores
        
    def p_rescore_with_combine_code(self, condense_query: str, original_query: str, topk_pids, original_scores, rerank_top_k=100):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # main_encoded = self.encode(original_query, full_length_search=False).to(device)
        
        combined_query = condense_query + " " + original_query
        combined_encoded = self.encode(combined_query, full_length_search=False).to(device)

        condense_query_tokens = self.checkpoint.query_tokenizer.tokenize([condense_query])
        split_point = len(condense_query_tokens[0])

        if split_point > combined_encoded.size(1):
            split_point = combined_encoded.size(1)

        if split_point == combined_encoded.size(1):
            main_encoded = combined_encoded
        else:
            main_encoded = combined_encoded[:, split_point:]

        print(f"Combined Query 2: {combined_query}")
        print(f"Split Point 2: {split_point}")

        Q = main_encoded[0].unsqueeze(0)
        
        
        
        doc_maxlen = self.config.doc_maxlen
        D_padded_result = self.checkpoint.docFromText([self.collection[pid] for pid in topk_pids], bsize=32, keep_dims=True)

        D_padded = D_padded_result[0] if isinstance(D_padded_result, tuple) else D_padded_result

        D_mask = torch.ones(D_padded.shape[:-1], dtype=torch.int, device=D_padded.device)
        
        
        positive_scores = colbert_score_original(main_encoded, D_padded, D_mask)


        sorted_indices = torch.argsort(positive_scores, descending=True).tolist()  # 降序排序
        sorted_pids = [topk_pids[i] for i in sorted_indices]
        sorted_scores = positive_scores[sorted_indices].tolist()


        return sorted_pids, sorted_scores
    

    def coverage_dense_search_for_batch(self, Q: torch.Tensor, k, threshold, boost_factor,filter_fn=None, pids=None):
        if k <= 10:
            if self.config.ncells is None:
                self.configure(ncells=1)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.5)
            if self.config.ndocs is None:
                self.configure(ndocs=256)
        elif k <= 100:
            if self.config.ncells is None:
                self.configure(ncells=2)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.45)
            if self.config.ndocs is None:
                self.configure(ndocs=1024)
        else:
            if self.config.ncells is None:
                self.configure(ncells=4)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.4)
            if self.config.ndocs is None:
                self.configure(ndocs=max(k * 4, 4096))
        
        pids, scores = self.ranker.coverage_rank_for_batch(self.config, Q, threshold, boost_factor, filter_fn, pids)

        return pids[:k], list(range(1, k+1)), scores[:k]
    
    def coverage_search_for_batch(self, condense_query: str, rewrite_text: str, k, threshold, boost_factor,combine_order='original_first'):
        # Concatenate condense query and rewrite query
        
        if combine_order == 'condense_first':
            combined_query = condense_query + " " + rewrite_text
            condense_query_tokens = self.checkpoint.query_tokenizer.tokenize([rewrite_text])
        else:
            combined_query = rewrite_text + " " + condense_query
            condense_query_tokens = self.checkpoint.query_tokenizer.tokenize([condense_query])

        # Encode the combined query
        combined_encoded = self.encode(combined_query, full_length_search=False)
        # Ensure condense_query and rewrite_text are lists for tokenization
        split_point = len(condense_query_tokens[0])

        # Ensure split_point does not exceed the length of combined_encoded
        if split_point > combined_encoded.size(1):
            split_point = combined_encoded.size(1)

        # If split_point is equal to the length, use the entire encoded vector
        if split_point == combined_encoded.size(1):
            rewrite_encoded = combined_encoded
        else:
            # Extract only the rewrite part of the encoded vector
            rewrite_encoded = combined_encoded[:, split_point:]
        # rewrite_encoded = self.encode(rewrite_text)

        # Perform dense search using only the rewrite_encoded vector
        return self.coverage_dense_search_for_batch(rewrite_encoded, k, threshold, boost_factor)
        # return self.dense_search(rewrite_encoded, k)

    
    