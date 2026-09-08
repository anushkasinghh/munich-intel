"""Tests for the retrieval eval's question loader.

Covers the rules that decide how a question is scored — which is where a silent
mistake would be most expensive, since a misclassified question still produces a
plausible-looking number.
"""

import pytest
import yaml

from munich_intel.eval.qa_dataset import (
    QUESTIONS_PATH,
    EvalQuestion,
    labelled,
    load_questions,
    negative_controls,
    recall_scored,
    unlabelled,
)


def question(**overrides) -> EvalQuestion:
    return EvalQuestion(**{"id": "q", "question": "?", "kind": "single-company", **overrides})


class TestExpectsEmpty:
    def test_a_negative_control_is_labelled_without_urls(self):
        # Nothing left for a human to fill in: having no answer IS the answer.
        assert question(expects_empty=True).is_labelled is True

    def test_a_negative_control_does_not_score_recall(self):
        # Folding it into recall would drag every pooled rate toward zero and make
        # a genuine improvement read as a regression.
        assert question(expects_empty=True).scores_recall is False

    def test_an_ordinary_unlabelled_question_is_not_labelled(self):
        assert question().is_labelled is False

    def test_claiming_both_is_rejected(self):
        with pytest.raises(ValueError, match="cannot be both"):
            question(expects_empty=True, relevant_urls=["https://example.com"])


class TestScoresDiversity:
    @pytest.mark.parametrize("kind", ["multi-company", "comparison"])
    def test_kinds_where_several_companies_should_appear(self, kind):
        assert question(kind=kind, relevant_urls=["https://a.example"]).scores_diversity is True

    @pytest.mark.parametrize("kind", ["single-company", "aggregation"])
    def test_kinds_where_diversity_is_not_a_goal(self, kind):
        # "What is Konux?" ideally returns one company. Counting diversity here and
        # averaging it with multi-company questions would produce a headline that
        # rises when single-company answers get worse.
        assert question(kind=kind, relevant_urls=["https://a.example"]).scores_diversity is False

    def test_a_negative_control_is_excluded_even_when_its_kind_qualifies(self):
        assert question(kind="multi-company", expects_empty=True).scores_diversity is False


class TestGrouping:
    def _set(self) -> list[EvalQuestion]:
        return [
            question(id="labelled", relevant_urls=["https://a.example"]),
            question(id="control", kind="multi-company", expects_empty=True),
            question(id="todo"),
        ]

    def test_labelled_includes_controls_but_not_the_backlog(self):
        assert [q.id for q in labelled(self._set())] == ["labelled", "control"]

    def test_recall_scored_excludes_controls(self):
        assert [q.id for q in recall_scored(self._set())] == ["labelled"]

    def test_negative_controls_are_isolated(self):
        assert [q.id for q in negative_controls(self._set())] == ["control"]

    def test_backlog_is_only_the_genuinely_unlabelled(self):
        assert [q.id for q in unlabelled(self._set())] == ["todo"]


class TestLoadQuestions:
    def test_duplicate_ids_are_rejected(self, tmp_path):
        # Ids identify a question across runs and in EVAL_DESIGN.md's delta tables,
        # so a shared id would make a reported before/after ambiguous.
        path = tmp_path / "questions.yaml"
        path.write_text(
            yaml.safe_dump(
                [
                    {"id": "dup", "question": "a?", "kind": "single-company"},
                    {"id": "dup", "question": "b?", "kind": "comparison"},
                ]
            )
        )
        with pytest.raises(ValueError, match="duplicate question ids"):
            load_questions(path)

    def test_an_invented_kind_is_rejected(self, tmp_path):
        path = tmp_path / "questions.yaml"
        path.write_text(yaml.safe_dump([{"id": "q", "question": "a?", "kind": "trivia"}]))
        with pytest.raises(ValueError):
            load_questions(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_questions(tmp_path / "nope.yaml")


class TestRealQuestionSet:
    """The checked-in set must stay loadable — it is edited by hand between runs."""

    def test_it_parses(self):
        assert load_questions(QUESTIONS_PATH)

    def test_every_negative_control_is_a_multi_company_question(self):
        # A single-company control would be incoherent: it would name a company the
        # corpus does have, and the "unanswerable" part would be the topic, not the
        # company. Keeping controls multi-company keeps them about the corpus.
        for control in negative_controls(load_questions(QUESTIONS_PATH)):
            assert control.kind == "multi-company"
