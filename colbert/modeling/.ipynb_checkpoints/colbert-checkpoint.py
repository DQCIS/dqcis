from colbert.infra.config.config import ColBERTConfig
from colbert.search.strided_tensor import StridedTensor
from colbert.utils.utils import print_message, flatten
from colbert.modeling.base_colbert import BaseColBERT
from colbert.parameters import DEVICE

import torch
import string

import os
import pathlib
from torch.utils.cpp_extension import load


class ColBERT(BaseColBERT):
    """
        This class handles the basic encoding and scoring operations in ColBERT. It is used for training.
    """

    def __init__(self, name='bert-base-uncased', colbert_config=None):
        super().__init__(name, colbert_config)
        self.use_gpu = colbert_config.total_visible_gpus > 0

        ColBERT.try_load_torch_extensions(self.use_gpu)

        if self.colbert_config.mask_punctuation:
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.raw_tokenizer.encode(symbol, add_special_tokens=False)[0]]}
        self.pad_token = self.raw_tokenizer.pad_token_id


    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or use_gpu:
            return

        print_message(f"Loading segmented_maxsim_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)...")
        segmented_maxsim_cpp = load(
            name="segmented_maxsim_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "segmented_maxsim.cpp"
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.segmented_maxsim = segmented_maxsim_cpp.segmented_maxsim_cpp

        cls.loaded_extensions = True

    def forward(self, Q, D):
        Q = self.query(*Q)
        D, D_mask = self.doc(*D, keep_dims='return_mask')

        # Repeat each query encoding for every corresponding document.
        Q_duplicated = Q.repeat_interleave(self.colbert_config.nway, dim=0).contiguous()
        scores = self.score(Q_duplicated, D, D_mask)

        if self.colbert_config.use_ib_negatives:
            ib_loss = self.compute_ib_loss(Q, D, D_mask)
            return scores, ib_loss

        return scores

    def compute_ib_loss(self, Q, D, D_mask):
        # TODO: Organize the code below! Quite messy.
        scores = (D.unsqueeze(0) @ Q.permute(0, 2, 1).unsqueeze(1)).flatten(0, 1)  # query-major unsqueeze

        scores = colbert_score_reduce(scores, D_mask.repeat(Q.size(0), 1, 1), self.colbert_config)

        nway = self.colbert_config.nway
        all_except_self_negatives = [list(range(qidx*D.size(0), qidx*D.size(0) + nway*qidx+1)) +
                                     list(range(qidx*D.size(0) + nway * (qidx+1), qidx*D.size(0) + D.size(0)))
                                     for qidx in range(Q.size(0))]

        scores = scores[flatten(all_except_self_negatives)]
        scores = scores.view(Q.size(0), -1)  # D.size(0) - self.colbert_config.nway + 1)

        labels = torch.arange(0, Q.size(0), device=scores.device) * (self.colbert_config.nway)

        return torch.nn.CrossEntropyLoss()(scores, labels)

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        assert keep_dims in [True, False, 'return_mask']

        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)
        mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)
        if self.use_gpu:
            D = D.half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == 'return_mask':
            return D, mask.bool()

        return D

    def score(self, Q, D_padded, D_mask):
        # assert self.colbert_config.similarity == 'cosine'
        if self.colbert_config.similarity == 'l2':
            assert self.colbert_config.interaction == 'colbert'
            return (-1.0 * ((Q.unsqueeze(2) - D_padded.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)
        return colbert_score(Q, D_padded, D_mask, config=self.colbert_config)

    def mask(self, input_ids, skiplist):
        mask = [[(x not in skiplist) and (x != self.pad_token) for x in d] for d in input_ids.cpu().tolist()]
        return mask


# TODO: In Query/DocTokenizer, use colbert.raw_tokenizer

# TODO: The masking below might also be applicable in the kNN part
def colbert_score_reduce(scores_padded, D_mask, config: ColBERTConfig):
    D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    scores_padded[D_padding] = -9999
    scores = scores_padded.max(1).values

    assert config.interaction in ['colbert', 'flipr'], config.interaction

    if config.interaction == 'flipr':
        assert config.query_maxlen == 64, ("for now", config)
        # assert scores.size(1) == config.query_maxlen, scores.size()

        K1 = config.query_maxlen // 2
        K2 = 8

        A = scores[:, :config.query_maxlen].topk(K1, dim=-1).values.sum(-1)
        B = 0

        if K2 <= scores.size(1) - config.query_maxlen:
            B = scores[:, config.query_maxlen:].topk(K2, dim=-1).values.sum(1)

        return A + B

    return scores.sum(-1)


# TODO: Wherever this is called, pass `config=`
def colbert_score(Q, D_padded, D_mask, config=ColBERTConfig()):
    """
        Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
        If Q.size(0) is 1, the matrix will be compared with all passages.
        Otherwise, each query matrix will be compared against the *aligned* passage.

        EVENTUALLY: Consider masking with -inf for the maxsim (or enforcing a ReLU).
    """

    use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, Q.size()
    assert D_padded.dim() == 3, D_padded.size()
    assert Q.size(0) in [1, D_padded.size(0)]

    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)

    return colbert_score_reduce(scores, D_mask, config)


def colbert_score_packed(Q, D_packed, D_lengths, config=ColBERTConfig()):
    """
        Works with a single query only.
    """

    use_gpu = config.total_visible_gpus > 0

    if use_gpu:
        Q, D_packed, D_lengths = Q.cuda(), D_packed.cuda(), D_lengths.cuda()

    Q = Q.squeeze(0)

    assert Q.dim() == 2, Q.size()
    assert D_packed.dim() == 2, D_packed.size()

    scores = D_packed @ Q.to(dtype=D_packed.dtype).T

    if use_gpu or config.interaction == "flipr":
        scores_padded, scores_mask = StridedTensor(scores, D_lengths, use_gpu=use_gpu).as_padded_tensor()

        return colbert_score_reduce(scores_padded, scores_mask, config)
    else:
        return ColBERT.segmented_maxsim(scores, D_lengths)

    
    

def colbert_score_rescore(Q, D_padded, D_mask, config=ColBERTConfig()):
    use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, Q.size()
    assert D_padded.dim() == 3, D_padded.size()
    assert Q.size(0) in [1, D_padded.size(0)]

    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)

    return colbert_score_reduce_rescore(scores, D_mask, config)

def colbert_score_reduce_rescore(scores_padded, D_mask, config, threshold=0.3, penalty_factor=0.05):
    D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1))
    scores_padded[D_padding] = -9999
    scores = scores_padded.max(2).values


    relevant_scores = scores[scores > threshold]
    
    relevant_count = relevant_scores.size(0)
    relevant_sum = relevant_scores.sum()


    irrelevant_high_scores = scores[(scores <= threshold) & (scores > threshold/2)]
    penalty = penalty_factor * irrelevant_high_scores.size(0)


    final_score = relevant_sum * (relevant_count / scores.numel()) - penalty

    return final_score

def colbert_score_positive(Q, D_padded, D_mask, config=ColBERTConfig()):

    use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, f"Q dim is {Q.dim()}, expected 3"
    assert D_padded.dim() == 3, f"D_padded dim is {D_padded.dim()}, expected 3"
    assert Q.size(0) in [1, D_padded.size(0)], f"Q size(0) is {Q.size(0)}, D_padded size(0) is {D_padded.size(0)}"

    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)
    

    D_mask = D_mask.view(scores.size(0), scores.size(1))  # Ensure D_mask shape matches scores shape
    

    print(f"D_mask shape after view: {D_mask.shape}")

    D_padding = ~D_mask.bool()
    scores[D_padding] = -9999
    scores = scores.max(1).values

    return scores.mean(-1)

def colbert_score_negative(Q, D_padded, D_mask, top_k=5, config=ColBERTConfig()):

    use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, f"Q dim is {Q.dim()}, expected 3"
    assert D_padded.dim() == 3, f"D_padded dim is {D_padded.dim()}, expected 3"
    assert Q.size(0) in [1, D_padded.size(0)], f"Q size(0) is {Q.size(0)}, D_padded size(0) is {D_padded.size(0)}"

    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)
    

    D_mask = D_mask.view(scores.size(0), scores.size(1))  # Ensure D_mask shape matches scores shape
    

    print(f"D_mask shape after view: {D_mask.shape}")

    D_padding = ~D_mask.bool()
    scores[D_padding] = -9999
    scores = scores.max(1).values


    top_scores = scores.topk(top_k, dim=-1).values

    return top_scores.sum(-1)


def improved_colbert_score(Q, D_padded, D_mask, k=3, min_score=0.5, balance_factor=0.5, config=ColBERTConfig()):

    use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, f"Q dim is {Q.dim()}, expected 3"
    assert D_padded.dim() == 3, f"D_padded dim is {D_padded.dim()}, expected 3"
    assert Q.size(0) in [1, D_padded.size(0)], f"Q size(0) is {Q.size(0)}, D_padded size(0) is {D_padded.size(0)}"

    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)
    
    D_mask = D_mask.view(scores.size(0), scores.size(1))
    print(f"D_mask shape after view: {D_mask.shape}")

    D_padding = ~D_mask.bool()
    scores[D_padding] = -9999


    doc_lengths = D_mask.sum(dim=1)
    query_length = Q.size(1)

    k_values = torch.clamp(torch.min(doc_lengths, torch.tensor(query_length)), max=k)

    top_k_scores = torch.stack([
        scores[i, :, j].topk(k_values[i], dim=0).values
        for i in range(scores.size(0))
        for j in range(scores.size(2))
    ]).view(scores.size(0), scores.size(2), -1)

    avg_scores = top_k_scores.mean(dim=2)
    match_counts = (top_k_scores > min_score).float().sum(dim=2)

    balance_score = 1 - (match_counts.max(dim=1).values - match_counts.min(dim=1).values) / (match_counts.sum(dim=1) + 1e-6)

    quality_score = avg_scores.mean(dim=1)

    min_query_score = avg_scores.min(dim=1).values

    final_score = quality_score * (1 - balance_factor + balance_factor * balance_score)

    final_score = torch.where(min_query_score < min_score, final_score * (min_query_score / min_score), final_score)

    return final_score


def colbert_score_original_with_flipr(Q, D_padded, D_mask, config=ColBERTConfig()):

    use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, f"Q dim is {Q.dim()}, expected 3"
    assert D_padded.dim() == 3, f"D_padded dim is {D_padded.dim()}, expected 3"
    assert Q.size(0) in [1, D_padded.size(0)], f"Q size(0) is {Q.size(0)}, D_padded size(0) is {D_padded.size(0)}"

    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)
    
    D_mask = D_mask.view(scores.size(0), scores.size(1))
    print(f"D_mask shape after view: {D_mask.shape}")

    D_padding = ~D_mask.bool()
    scores[D_padding] = -9999
    scores = scores.max(1).values

    if config.interaction == 'flipr':
        assert config.query_maxlen == 64, ("for now", config)
        
        K1 = config.query_maxlen // 2
        K2 = 8

        A = scores[:, :config.query_maxlen].topk(K1, dim=-1).values.sum(-1)
        B = 0

        if K2 <= scores.size(1) - config.query_maxlen:
            B = scores[:, config.query_maxlen:].topk(K2, dim=-1).values.sum(1)

        return A + B
    else:  # 'colbert'
        return scores.sum(-1)
    

def colbert_score_original(Q, D_padded, D_mask, config=ColBERTConfig()):

    use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, f"Q dim is {Q.dim()}, expected 3"
    assert D_padded.dim() == 3, f"D_padded dim is {D_padded.dim()}, expected 3"
    assert Q.size(0) in [1, D_padded.size(0)], f"Q size(0) is {Q.size(0)}, D_padded size(0) is {D_padded.size(0)}"

    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)
    
    D_mask = D_mask.view(scores.size(0), scores.size(1))
    print(f"D_mask shape after view: {D_mask.shape}")

    D_padding = ~D_mask.bool()
    scores[D_padding] = -9999
    scores = scores.max(1).values

    return scores.sum(-1)

def colbert_score_coverage(Q, D_padded, D_mask, threshold, config=ColBERTConfig()):

    use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, f"Q dim is {Q.dim()}, expected 3"
    assert D_padded.dim() == 3, f"D_padded dim is {D_padded.dim()}, expected 3"
    assert Q.size(0) in [1, D_padded.size(0)], f"Q size(0) is {Q.size(0)}, D_padded size(0) is {D_padded.size(0)}"


    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)

    D_mask = D_mask.view(scores.size(0), scores.size(1))
    print(f"D_mask shape after view: {D_mask.shape}")

    D_padding = ~D_mask.bool()
    scores[D_padding] = -9999
    
    scores = scores.max(1).values
    
    max_scores = scores.max(0).values
    
    matched_query_terms = (max_scores > threshold).float().sum()
    
    original_score = scores.sum(-1)
    
    final_score = original_score * (1 + 0.1 * matched_query_terms)
    
    return final_score

def colbert_score_sequence_aware(Q, D_padded, D_mask, config=ColBERTConfig()):

    use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, f"Q dim is {Q.dim()}, expected 3"
    assert D_padded.dim() == 3, f"D_padded dim is {D_padded.dim()}, expected 3"
    assert Q.size(0) in [1, D_padded.size(0)], f"Q size(0) is {Q.size(0)}, D_padded size(0) is {D_padded.size(0)}"


    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)

    D_mask = D_mask.view(scores.size(0), scores.size(1))
    print(f"D_mask shape after view: {D_mask.shape}")

    D_padding = ~D_mask.bool()
    scores[D_padding] = -9999

    best_match = scores.max(2).indices
    

    def lcs_length(seq):
        n = seq.size(0)
        lengths = torch.zeros(n+1, n+1, device=seq.device)
        for i in range(1, n+1):
            for j in range(1, n+1):
                if seq[i-1] == seq[j-1]:
                    lengths[i,j] = lengths[i-1,j-1] + 1
                else:
                    lengths[i,j] = max(lengths[i-1,j], lengths[i,j-1])
        return lengths[n,n]
    
    lcs_scores = torch.tensor([lcs_length(bm) for bm in best_match], device=scores.device)
    

    original_scores = scores.max(1).values.sum(-1)
    

    final_score = original_scores * (1 + 0.1 * lcs_scores)
    
    return final_score


def colbert_score_coverage_for_batch(Q, D_padded, D_mask, threshold, boost_factor, config=ColBERTConfig()):
    use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()
    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)
    D_mask = D_mask.view(scores.size(0), scores.size(1))
    D_padding = ~D_mask.bool()
    scores[D_padding] = -9999
    scores = scores.max(1).values
    max_scores = scores.max(0).values
    matched_query_terms = (max_scores > threshold).float().sum()
    original_score = scores.sum(-1)
    final_score = original_score * (1 + boost_factor * matched_query_terms)
    return final_score

def colbert_score_reduce_for_coverage(scores_padded, D_mask, config: ColBERTConfig, threshold, boost_factor):
    D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    scores_padded[D_padding] = -9999
    scores = scores_padded.max(1).values
    
    original_score = scores.sum(-1)


    max_scores = scores.max(0).values
    # print(f"max_scores {max_scores}")
    matched_query_terms = (max_scores > threshold).float().sum()
    # print(f"matched_query_terms {matched_query_terms}")
    

    final_score = original_score * (1 + boost_factor * matched_query_terms)

    return final_score

def colbert_score_for_coverage(Q, D_padded, D_mask, threshold, boost_factor, config=ColBERTConfig()):
    use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, Q.size()
    assert D_padded.dim() == 3, D_padded.size()
    assert Q.size(0) in [1, D_padded.size(0)]

    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)

    return colbert_score_reduce(scores, D_mask, config, threshold, boost_factor)

def colbert_score_packed_for_coverage(Q, D_packed, D_lengths, threshold, boost_factor, config=ColBERTConfig()):
    use_gpu = config.total_visible_gpus > 0

    if use_gpu:
        Q, D_packed, D_lengths = Q.cuda(), D_packed.cuda(), D_lengths.cuda()

    Q = Q.squeeze(0)

    assert Q.dim() == 2, Q.size()
    assert D_packed.dim() == 2, D_packed.size()

    scores = D_packed @ Q.to(dtype=D_packed.dtype).T

    if use_gpu or config.interaction == "flipr":
        scores_padded, scores_mask = StridedTensor(scores, D_lengths, use_gpu=use_gpu).as_padded_tensor()

        return colbert_score_reduce_for_coverage(scores_padded, scores_mask, config, threshold, boost_factor)
    else:
        return ColBERT.segmented_maxsim(scores, D_lengths)