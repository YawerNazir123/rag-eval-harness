"""Retrieval and answer metrics for RAG evaluation.

Every function here is pure: it takes plain values and returns a float
or a bool, with no I/O and no model calls. That makes the whole library
unit-testable with hand-computed expected values.
"""

from __future__ import annotations

import math
import re


def recall_at_k(retrieved: list[int], relevant: list[int] | set[int], k: int) -> float:
    """Fraction of relevant chunk ids found in the top ``k`` retrieved ids.

    Returns 0.0 when there are no relevant ids (nothing to recall).
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant_set) / len(relevant_set)


def hit_rate(retrieved: list[int], relevant: list[int] | set[int], k: int) -> float:
    """1.0 if at least one relevant chunk id is in the top ``k``, else 0.0."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return 1.0 if set(retrieved[:k]) & relevant_set else 0.0


def reciprocal_rank(retrieved: list[int], relevant: list[int] | set[int]) -> float:
    """Reciprocal rank of the first relevant chunk id, 0.0 if none found."""
    relevant_set = set(relevant)
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0


def mrr(retrieved: list[int], relevant: list[int] | set[int]) -> float:
    """Mean reciprocal rank for a single query (alias of reciprocal_rank)."""
    return reciprocal_rank(retrieved, relevant)


def ndcg_at_k(retrieved: list[int], relevant: list[int] | set[int], k: int) -> float:
    """Normalised discounted cumulative gain at ``k`` (binary relevance).

    DCG sums 1 / log2(rank + 1) over relevant ids in the top ``k``;
    IDCG is the DCG of the ideal ranking. Returns 0.0 when there are
    no relevant ids.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = retrieved[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(top_k, start=1)
        if chunk_id in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return _normalize(text).split() if _normalize(text) else []


def token_f1(predicted: str, reference: str) -> float:
    """Token-level F1 between a predicted answer and a reference answer.

    Precision and recall are computed over multisets of tokens, so
    repeated words count. Returns 0.0 when either side has no tokens,
    except that two empty strings score 1.0.
    """
    pred_tokens = _tokens(predicted)
    ref_tokens = _tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    from collections import Counter

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(predicted: str, reference: str) -> float:
    """1.0 when the normalised predicted and reference answers are equal."""
    return 1.0 if _normalize(predicted) == _normalize(reference) else 0.0


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for i, token_a in enumerate(a, start=1):
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[len(b)]


def rouge_l(predicted: str, reference: str) -> float:
    """ROUGE-L F1 from the longest common subsequence of tokens.

    Implemented here with a plain LCS so the harness needs no heavy
    dependency. Returns 0.0 when either side has no tokens, except
    that two empty strings score 1.0.
    """
    pred_tokens = _tokens(predicted)
    ref_tokens = _tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def citation_correct(predicted_species: str | None, expected_species: str | None) -> bool:
    """True when the cited species matches the expected species.

    Comparison is case-insensitive and ignores surrounding whitespace.
    A missing citation (None) is never correct when an expected species
    exists; when both are None the citation is vacuously correct.
    """
    if expected_species is None:
        return predicted_species is None
    if predicted_species is None:
        return False
    return predicted_species.strip().lower() == expected_species.strip().lower()


def abstention_correct(predicted_abstained: bool, should_abstain: bool) -> bool:
    """True when the system abstained exactly when it should have."""
    return predicted_abstained == should_abstain


def mean(values: list[float]) -> float:
    """Arithmetic mean, 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0
