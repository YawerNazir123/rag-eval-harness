"""Unit tests for the metric library, with hand-computed expected values."""

import pytest

from rag_eval.metrics import (
    abstention_correct,
    citation_correct,
    exact_match,
    hit_rate,
    mrr,
    ndcg_at_k,
    recall_at_k,
    rouge_l,
    token_f1,
)


def test_recall_at_k_partial():
    # Top 2 of [3, 1, 7] contains 1 of the 2 relevant ids {1, 5}.
    assert recall_at_k([3, 1, 7], [1, 5], k=2) == pytest.approx(0.5)
    assert recall_at_k([3, 1, 7], [1, 5], k=5) == pytest.approx(0.5)


def test_recall_at_k_all_and_none():
    assert recall_at_k([1, 5], [1, 5], k=5) == pytest.approx(1.0)
    assert recall_at_k([2, 3], [1, 5], k=5) == pytest.approx(0.0)
    assert recall_at_k([1], [], k=5) == pytest.approx(0.0)


def test_hit_rate():
    assert hit_rate([3, 1, 7], [1, 5], k=2) == pytest.approx(1.0)
    assert hit_rate([3, 7], [1, 5], k=5) == pytest.approx(0.0)
    assert hit_rate([1], [], k=5) == pytest.approx(0.0)


def test_mrr():
    # First relevant id at rank 2 -> 1/2.
    assert mrr([3, 1, 7], [1]) == pytest.approx(0.5)
    assert mrr([1, 2], [1]) == pytest.approx(1.0)
    assert mrr([2, 3], [1]) == pytest.approx(0.0)
    # First of several relevant ids counts: rank 3 -> 1/3.
    assert mrr([2, 3, 5], [5, 9]) == pytest.approx(1 / 3)


def test_ndcg_at_k_perfect():
    assert ndcg_at_k([1], [1], k=5) == pytest.approx(1.0)
    assert ndcg_at_k([1, 2, 3], [1, 2, 3], k=3) == pytest.approx(1.0)


def test_ndcg_at_k_partial():
    # Retrieved [1, 2, 3], relevant {1, 3}, k=3:
    # DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093
    # nDCG = 1.5 / 1.63093 = 0.91972
    assert ndcg_at_k([1, 2, 3], [1, 3], k=3) == pytest.approx(0.91972, abs=1e-4)


def test_ndcg_at_k_rank_matters():
    # Relevant id at rank 2: DCG = 1/log2(3) = 0.63093, IDCG = 1.
    assert ndcg_at_k([2, 1], [1], k=2) == pytest.approx(0.63093, abs=1e-4)
    assert ndcg_at_k([2, 3], [1], k=2) == pytest.approx(0.0)


def test_token_f1():
    assert token_f1("Kashmiri name Zanda tsar", "Kashmiri name Zanda tsar") == pytest.approx(1.0)
    # Overlap 2 tokens: precision 2/3, recall 2/2 -> F1 = 0.8.
    assert token_f1("the cat sat", "the cat") == pytest.approx(0.8)
    assert token_f1("dog", "cat") == pytest.approx(0.0)
    assert token_f1("", "") == pytest.approx(1.0)
    assert token_f1("something", "") == pytest.approx(0.0)


def test_exact_match():
    assert exact_match("Hello, World!", "hello world") == pytest.approx(1.0)
    assert exact_match("Kashmiri name: Ans", "Kashmiri name Ans") == pytest.approx(1.0)
    assert exact_match("Ans", "Zanda tsar") == pytest.approx(0.0)


def test_rouge_l():
    assert rouge_l("the cat sat", "the cat sat") == pytest.approx(1.0)
    # LCS "the cat sat" = 3 tokens; precision 3/6, recall 3/3 -> F1 = 2/3.
    assert rouge_l("the cat sat on the mat", "the cat sat") == pytest.approx(2 / 3)
    # Word order differs: LCS is 1 token ("b" or "a" alone), not 2.
    assert rouge_l("a b", "b a") == pytest.approx(0.5)
    assert rouge_l("dog", "cat") == pytest.approx(0.0)
    assert rouge_l("", "") == pytest.approx(1.0)


def test_citation_correct():
    assert citation_correct("Black Kite", "black kite") is True
    assert citation_correct(" Black Kite ", "Black Kite") is True
    assert citation_correct(None, "Black Kite") is False
    assert citation_correct("Barn Owl", "Black Kite") is False
    assert citation_correct(None, None) is True
    assert citation_correct("Barn Owl", None) is False


def test_abstention_correct():
    assert abstention_correct(True, True) is True
    assert abstention_correct(False, False) is True
    assert abstention_correct(True, False) is False
    assert abstention_correct(False, True) is False
