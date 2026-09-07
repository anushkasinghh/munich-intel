"""Hand-computed examples for eval.metrics.

Deliberately no fixtures from real data. If these tests loaded data/entities/ they
would pass whenever the code was self-consistent — including when it was
self-consistently wrong. Every expected number below was worked out on paper first,
so the tests can disagree with the implementation.
"""

import pytest

from munich_intel.entities import JobPosting
from munich_intel.eval.metrics import (
    duplicate_url_groups,
    field_accuracy,
    field_comparable_count,
    match,
    normalize_url,
    prf1,
    prf1_from_counts,
)


def job(url: str, title: str = "Engineer", location: str | None = None) -> JobPosting:
    return JobPosting(
        company_slug="acme",
        title=title,
        url=url,
        location=location,
        scraped_at="2026-08-21",
    )


class TestNormalizeUrl:
    @pytest.mark.parametrize(
        "variant",
        [
            "https://acme.example.com/jobs/1",
            "https://acme.example.com/jobs/1/",  # trailing slash
            "https://ACME.example.COM/jobs/1",  # host case
            "https://acme.example.com/jobs/1?language=en",  # tracking query
            "https://acme.example.com/jobs/1#apply",  # fragment
            "  https://acme.example.com/jobs/1  ",  # stray whitespace
        ],
    )
    def test_variants_of_one_listing_collapse_to_one_key(self, variant):
        assert normalize_url(variant) == "https://acme.example.com/jobs/1"

    def test_path_case_is_preserved(self):
        # Hosts are case-insensitive, paths are not: /Jobs and /jobs may be
        # different pages, so normalizing them together could hide a scraper bug.
        assert normalize_url("https://acme.example.com/Jobs") != normalize_url(
            "https://acme.example.com/jobs"
        )

    def test_different_listings_stay_different(self):
        assert normalize_url("https://acme.example.com/jobs/1") != normalize_url(
            "https://acme.example.com/jobs/2"
        )


class TestPrf1:
    def test_worked_example(self):
        # predicted {a,b,c}, gold {b,c,d}
        #   tp = |{b,c}| = 2, fp = |{a}| = 1, fn = |{d}| = 1
        #   precision = 2/3, recall = 2/3, f1 = 2/3
        result = prf1({"a", "b", "c"}, {"b", "c", "d"})
        assert (result.tp, result.fp, result.fn) == (2, 1, 1)
        assert result.precision == pytest.approx(2 / 3)
        assert result.recall == pytest.approx(2 / 3)
        assert result.f1 == pytest.approx(2 / 3)

    def test_lopsided_scores_are_punished_by_f1(self):
        # predicted {a}, gold {a,b,c,d}: perfect precision, recall 1/4
        #   f1 = 2 * 1 * 0.25 / 1.25 = 0.4 — nowhere near the 0.625 an average gives
        result = prf1({"a"}, {"a", "b", "c", "d"})
        assert result.precision == 1.0
        assert result.recall == pytest.approx(0.25)
        assert result.f1 == pytest.approx(0.4)

    def test_both_empty_is_a_correct_answer(self):
        result = prf1(set(), set())
        assert (result.precision, result.recall, result.f1) == (1.0, 1.0, 1.0)
        assert (result.tp, result.fp, result.fn) == (0, 0, 0)

    def test_predicted_empty_against_real_gold_scores_zero(self):
        result = prf1(set(), {"a", "b"})
        assert (result.precision, result.recall, result.f1) == (0.0, 0.0, 0.0)
        assert (result.tp, result.fp, result.fn) == (0, 0, 2)

    def test_gold_empty_against_predictions_scores_zero_not_nan(self):
        # Every prediction is a false positive and there is nothing to recall.
        # Recall's denominator is 0 — the number that must not come back NaN.
        result = prf1({"a", "b"}, set())
        assert (result.precision, result.recall, result.f1) == (0.0, 0.0, 0.0)
        assert (result.tp, result.fp, result.fn) == (0, 2, 0)

    def test_counts_aggregate_across_companies(self):
        # Two companies: one with a single posting, found; one with 41 postings, of
        # which 1 was found. Averaging the two recall rates gives (1.0 + 1/41)/2 =
        # 0.51 and flatters the extractor badly; pooling the counts gives the honest
        # 2/42 = 0.048. This is why PRF1 carries tp/fp/fn and not just rates.
        small = prf1({"a"}, {"a"})
        large = prf1({"b"}, {f"g{i}" for i in range(40)} | {"b"})
        pooled_recall = (small.tp + large.tp) / (small.tp + large.tp + small.fn + large.fn)
        assert pooled_recall == pytest.approx(2 / 42)
        assert (small.recall + large.recall) / 2 == pytest.approx(0.5122, abs=1e-4)


class TestPrf1FromCounts:
    def test_agrees_with_the_set_based_prf1(self):
        # prf1 delegates here, so the two must never drift apart.
        assert prf1_from_counts(2, 1, 1) == prf1({"a", "b", "c"}, {"b", "c", "d"})

    def test_all_zero_counts_score_one(self):
        assert prf1_from_counts(0, 0, 0) == prf1(set(), set())

    def test_pooling_matches_a_hand_computed_aggregate(self):
        # The aggregate row in scripts/eval_jobs.py: 23 tp, 6 fp, 5 fn over 5 companies.
        # precision = 23/29 = 0.793, recall = 23/28 = 0.821
        pooled = prf1_from_counts(23, 6, 5)
        assert pooled.precision == pytest.approx(23 / 29)
        assert pooled.recall == pytest.approx(23 / 28)


class TestMatch:
    def test_splits_into_matched_fp_and_fn(self):
        predicted = [job("https://acme.example.com/jobs/1"), job("https://acme.example.com/bogus")]
        gold = [job("https://acme.example.com/jobs/1"), job("https://acme.example.com/jobs/2")]
        matched, false_positives, false_negatives = match(predicted, gold)

        assert len(matched) == 1
        assert str(matched[0][0].url) == "https://acme.example.com/jobs/1"
        assert [str(p.url) for p in false_positives] == ["https://acme.example.com/bogus"]
        assert [str(g.url) for g in false_negatives] == ["https://acme.example.com/jobs/2"]

    def test_matches_through_url_normalization(self):
        # Same listing, different tracking param — must not count as FP + FN.
        matched, false_positives, false_negatives = match(
            [job("https://acme.example.com/jobs/1?language=en")],
            [job("https://acme.example.com/jobs/1")],
        )
        assert len(matched) == 1
        assert false_positives == []
        assert false_negatives == []

    def test_postings_sharing_a_url_collapse_to_one(self):
        # The Isar/ClearOps failure mode: several "jobs" pointing at one careers
        # page. They are one URL key, so they score as one false positive.
        predicted = [
            job("https://acme.example.com/careers", title="Ready to join?"),
            job("https://acme.example.com/careers", title="Our hiring process"),
        ]
        matched, false_positives, _ = match(predicted, [])
        assert matched == []
        assert len(false_positives) == 1


class TestFieldAccuracy:
    def test_exact_match_over_matched_pairs(self):
        matched = [
            (job("https://x/1", title="ML Engineer"), job("https://x/1", title="ML Engineer")),
            (job("https://x/2", title="Ml engineer"), job("https://x/2", title="ML Engineer")),
        ]
        assert field_accuracy(matched, "title") == pytest.approx(0.5)

    def test_pairs_with_no_gold_value_are_skipped(self):
        # 3 pairs, only 2 with a gold location, 1 of those correct -> 1/2, not 1/3.
        matched = [
            (job("https://x/1", location="Munich"), job("https://x/1", location="Munich")),
            (job("https://x/2", location="Berlin"), job("https://x/2", location="Munich")),
            (job("https://x/3", location="Munich"), job("https://x/3", location=None)),
        ]
        assert field_accuracy(matched, "location") == pytest.approx(0.5)
        assert field_comparable_count(matched, "location") == 2

    def test_nothing_comparable_returns_zero_not_nan(self):
        matched = [(job("https://x/1", location="Munich"), job("https://x/1", location=None))]
        assert field_accuracy(matched, "location") == 0.0
        assert field_comparable_count(matched, "location") == 0

    def test_empty_matched_list(self):
        assert field_accuracy([], "title") == 0.0
        assert field_comparable_count([], "title") == 0


class TestDuplicateUrlGroups:
    def test_reports_only_groups_of_two_or_more(self):
        postings = [
            job("https://acme.example.com/careers", title="Ready to join?"),
            job("https://acme.example.com/careers/", title="Our process"),  # normalizes equal
            job("https://acme.example.com/jobs/1", title="ML Engineer"),
        ]
        groups = duplicate_url_groups(postings)
        assert list(groups) == ["https://acme.example.com/careers"]
        assert len(groups["https://acme.example.com/careers"]) == 2

    def test_all_unique_urls_report_nothing(self):
        assert duplicate_url_groups([job("https://x/1"), job("https://x/2")]) == {}
