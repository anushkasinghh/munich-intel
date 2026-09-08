"""Hand-computed examples for eval.retrieval_metrics.

Same rule as tests/test_metrics.py: no fixtures from Qdrant or data/raw/. Every
expected number below was worked out on paper first, so these tests are able to
disagree with the implementation instead of merely confirming it is self-consistent.

Several cases are written directly from the four measured failure modes in the
index, so if a later change fixes one of them the corresponding test reads as the
description of what "fixed" means.
"""

import pytest

from munich_intel.eval.retrieval_metrics import (
    RecallCounts,
    company_diversity,
    hit_rate,
    mrr,
    pooled_recall,
    recall_at_k,
    recall_counts_at_k,
    threshold_gap,
    top_score,
)

MARVEL = "https://marvelfusion.com/"
PROXIMA = "https://www.proximafusion.com/"
KONUX_NEWS = "https://news.google.com/rss/search?q=konux"
KONUX_SITE = "https://www.konux.com/company"


def hit(url: str, slug: str, score: float = 0.5) -> dict:
    return {"url": url, "company_slug": slug, "chunk_text": "...", "score": score}


class TestRecallAtK:
    def test_one_of_two_relevant_in_top_2(self):
        # Failure mode 2: "Compare Proxima and Marvel" returns two Proxima chunks.
        # Both companies are relevant, one is present -> 1/2.
        retrieved = [PROXIMA, PROXIMA]
        assert recall_at_k(retrieved, {PROXIMA, MARVEL}, k=2) == 0.5

    def test_duplicate_chunks_of_one_page_count_once(self):
        # Three chunks of the same page are one page found, not three.
        retrieved = [MARVEL, MARVEL, MARVEL]
        assert recall_at_k(retrieved, {MARVEL, PROXIMA}, k=3) == 0.5

    def test_relevant_below_the_cutoff_does_not_count(self):
        # Failure mode 3: konux.com/company sits at rank 3, outside k=2.
        retrieved = [KONUX_NEWS, KONUX_NEWS, KONUX_SITE]
        assert recall_at_k(retrieved, {KONUX_SITE}, k=2) == 0.0
        assert recall_at_k(retrieved, {KONUX_SITE}, k=3) == 1.0

    def test_recall_is_capped_when_relevant_exceeds_k(self):
        # Four relevant pages cannot be covered by two chunks: the ceiling is 0.5.
        relevant = {"https://a.example/1", "https://b.example/1", "https://c.example/1", MARVEL}
        retrieved = ["https://a.example/1", "https://b.example/1", "https://c.example/1"]
        assert recall_at_k(retrieved, relevant, k=2) == 0.5

    def test_k_larger_than_the_retrieved_list_is_fine(self):
        assert recall_at_k([MARVEL], {MARVEL}, k=10) == 1.0

    def test_nothing_relevant_returns_zero_not_nan(self):
        assert recall_at_k([KONUX_NEWS], {MARVEL}, k=2) == 0.0

    def test_empty_retrieved_list_returns_zero(self):
        assert recall_at_k([], {MARVEL}, k=2) == 0.0

    def test_urls_are_normalized_on_both_sides(self):
        # Trailing slash, host case and a tracking param must not split the key.
        assert recall_at_k(["https://MARVELFUSION.com/?utm_source=x"], {"https://marvelfusion.com"}, k=1) == 1.0

    def test_unlabelled_question_raises(self):
        with pytest.raises(ValueError, match="unlabelled"):
            recall_at_k([MARVEL], set(), k=2)

    @pytest.mark.parametrize("k", [0, -1])
    def test_nonsense_k_raises(self, k):
        with pytest.raises(ValueError, match="k must be"):
            recall_at_k([MARVEL], {MARVEL}, k=k)


class TestRecallCounts:
    def test_counts_match_the_rate(self):
        counts = recall_counts_at_k([PROXIMA, PROXIMA], {PROXIMA, MARVEL}, k=2)
        assert counts == RecallCounts(recall=0.5, found=1, total=2, k=2)


class TestMrr:
    def test_first_position_is_one(self):
        assert mrr([MARVEL, PROXIMA], {MARVEL}) == 1.0

    def test_third_position_is_one_third(self):
        # The konux case: recall@2 is 0 but the page is only one rank below.
        assert mrr([KONUX_NEWS, KONUX_NEWS, KONUX_SITE], {KONUX_SITE}) == pytest.approx(1 / 3)

    def test_uses_the_first_relevant_not_the_best(self):
        # Two relevant pages at ranks 2 and 3 -> 1/2, the earlier one wins.
        assert mrr([KONUX_NEWS, MARVEL, PROXIMA], {MARVEL, PROXIMA}) == 0.5

    def test_no_relevant_result_is_zero(self):
        assert mrr([KONUX_NEWS, KONUX_NEWS], {MARVEL}) == 0.0

    def test_is_not_truncated_at_k(self):
        # Rank 7 still scores — that is the point: it shows how far the cutoff missed.
        # Failure mode 1: proxima-fusion first appears at rank 7 behind six marvel chunks.
        retrieved = [MARVEL] * 6 + [PROXIMA]
        assert mrr(retrieved, {PROXIMA}) == pytest.approx(1 / 7)

    def test_empty_relevant_raises(self):
        with pytest.raises(ValueError, match="unlabelled"):
            mrr([MARVEL], set())


class TestCompanyDiversity:
    def test_monopolised_result_set_has_diversity_one(self):
        # Failure mode 1, stated as a number: six chunks, one company.
        retrieved = [hit(MARVEL, "marvel-fusion") for _ in range(6)]
        assert company_diversity(retrieved, k=6) == 1

    def test_counts_distinct_slugs_within_k_only(self):
        retrieved = [
            hit(MARVEL, "marvel-fusion"),
            hit(MARVEL, "marvel-fusion"),
            hit(PROXIMA, "proxima-fusion"),
        ]
        assert company_diversity(retrieved, k=2) == 1
        assert company_diversity(retrieved, k=3) == 2

    def test_different_urls_from_one_company_are_still_one_company(self):
        # A company's site page and its news page are two pages but one company.
        retrieved = [hit(KONUX_SITE, "konux"), hit(KONUX_NEWS, "konux")]
        assert company_diversity(retrieved, k=2) == 1

    def test_empty_result_set_is_zero(self):
        assert company_diversity([], k=5) == 0

    def test_missing_slug_raises_rather_than_undercounting(self):
        with pytest.raises(ValueError, match="company_slug"):
            company_diversity([{"url": MARVEL}], k=1)


class TestHitRate:
    def test_true_when_one_relevant_is_inside_k(self):
        assert hit_rate([KONUX_NEWS, KONUX_SITE], {KONUX_SITE}, k=2) is True

    def test_false_when_the_relevant_page_is_below_k(self):
        assert hit_rate([KONUX_NEWS, KONUX_NEWS, KONUX_SITE], {KONUX_SITE}, k=2) is False

    def test_a_half_answered_comparison_still_counts_as_a_hit(self):
        # Why hit_rate is reported next to recall rather than instead of it: this
        # scores 1.0 while the answer names only one of the two companies asked about.
        assert hit_rate([PROXIMA, PROXIMA], {PROXIMA, MARVEL}, k=2) is True
        assert recall_at_k([PROXIMA, PROXIMA], {PROXIMA, MARVEL}, k=2) == 0.5

    def test_empty_relevant_raises(self):
        with pytest.raises(ValueError, match="unlabelled"):
            hit_rate([MARVEL], set(), k=2)


class TestTopScore:
    def test_takes_the_best_not_the_first(self):
        retrieved = [hit(MARVEL, "marvel-fusion", 0.52), hit(PROXIMA, "proxima-fusion", 0.61)]
        assert top_score(retrieved) == 0.61

    def test_empty_result_set_is_zero(self):
        assert top_score([]) == 0.0


class TestThresholdGap:
    def test_positive_gap_means_a_cutoff_exists(self):
        # Every answerable question beats every unanswerable one, so a threshold
        # anywhere in the gap would correctly reject the unanswerable ones.
        assert threshold_gap([0.70, 0.80], [0.40, 0.55]) == pytest.approx(0.15)

    def test_overlap_means_no_cutoff_can_work(self):
        # Failure mode 4: the compressed 0.41-0.65 score band. The worst real
        # question (0.44) scores below the best junk one (0.63), so any cutoff that
        # rejected the junk would also throw away a genuine answer.
        assert threshold_gap([0.44, 0.65], [0.41, 0.63]) == pytest.approx(-0.19)

    def test_touching_distributions_give_no_room(self):
        # Exactly equal is still no gap: a cutoff needs somewhere to sit.
        assert threshold_gap([0.50], [0.50]) == 0.0

    def test_uses_worst_answerable_against_best_unanswerable(self):
        # Not the means: one answerable question scoring low is enough to close the
        # gap, because a threshold has to work for every question, not on average.
        assert threshold_gap([0.90, 0.90, 0.42], [0.60]) == pytest.approx(-0.18)

    @pytest.mark.parametrize("answerable,unanswerable", [([], [0.5]), ([0.5], []), ([], [])])
    def test_missing_either_side_demonstrates_nothing(self, answerable, unanswerable):
        # 0.0 rather than a positive number: with nothing to compare against, no
        # separation has been shown, and claiming one is the wrong way to be wrong.
        assert threshold_gap(answerable, unanswerable) == 0.0


class TestPooledRecall:
    def test_pools_counts_rather_than_averaging_rates(self):
        # One question: 1 of 1 found. Another: 1 of 8 found.
        # Averaging the rates gives (1.0 + 0.125) / 2 = 0.5625, which flatters a
        # system that found 2 of 9 pages. Pooling gives the honest 2/9 = 0.222.
        counts = [
            RecallCounts(recall=1.0, found=1, total=1, k=2),
            RecallCounts(recall=0.125, found=1, total=8, k=2),
        ]
        pooled = pooled_recall(counts)
        assert pooled.found == 2
        assert pooled.total == 9
        assert pooled.recall == pytest.approx(2 / 9)

    def test_empty_pool_is_zero_not_an_error(self):
        # A `kind` with no labelled questions is a legitimate empty table row.
        assert pooled_recall([]).recall == 0.0

    def test_mixing_different_k_raises(self):
        counts = [
            RecallCounts(recall=1.0, found=1, total=1, k=2),
            RecallCounts(recall=1.0, found=1, total=1, k=5),
        ]
        with pytest.raises(ValueError, match="different k"):
            pooled_recall(counts)
