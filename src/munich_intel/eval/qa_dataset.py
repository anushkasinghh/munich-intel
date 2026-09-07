"""Loading the retrieval eval's gold question set.

The retrieval counterpart to `datasets.py`, and the only module in the retrieval
eval that touches disk. Questions parse through a pydantic model for the same
reason the jobs gold parses through `JobPosting`: the file is hand-written, so a
typo'd field or an invented `kind` should fail loudly at load time rather than
score wrong in silence.

Labelling is hand work by design. `relevant_urls` is the answer key — which pages
in `data/raw/` genuinely answer the question — and a machine-labelled gold set
measures only whether retrieval agrees with itself.
"""

from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

QUESTIONS_PATH = Path("data/eval/questions.yaml")

QuestionKind = Literal["single-company", "multi-company", "comparison", "aggregation"]

# Aggregation questions ("how many companies are hiring?") are unanswerable by
# similarity search at any k: the answer is a count over the whole corpus, not a
# passage sitting in it. They are kept in the set and reported separately as
# expected failures — their job is to size the gap that the graph work later fills,
# not to be fixed by tuning retrieval. Excluding them would hide that gap; pooling
# them into the headline would make every retrieval change look worse than it is.
EXPECTED_TO_FAIL: frozenset[str] = frozenset({"aggregation"})


class EvalQuestion(BaseModel):
    """One labelled question. `relevant_urls` empty means "not labelled yet"."""

    id: str
    question: str
    kind: QuestionKind
    relevant_urls: list[str] = Field(default_factory=list)
    notes: str = ""

    @property
    def is_labelled(self) -> bool:
        return bool(self.relevant_urls)

    @property
    def expected_to_fail(self) -> bool:
        return self.kind in EXPECTED_TO_FAIL

    @property
    def relevant(self) -> set[str]:
        """The answer key as a set, for handing to `retrieval_metrics`.

        Left un-normalized here on purpose — the metrics normalize on the way in, so
        normalization lives in exactly one place and a change to `normalize_url`
        cannot leave the loader and the scorer disagreeing.
        """
        return set(self.relevant_urls)


def load_questions(path: Path = QUESTIONS_PATH) -> list[EvalQuestion]:
    """Parse data/eval/questions.yaml, rejecting duplicate ids.

    Ids are how a question is referred to across runs and in EVAL_DESIGN.md's delta
    tables, so two questions sharing one id would make a reported before/after
    ambiguous. Cheaper to catch here than to puzzle over later.
    """
    if not path.exists():
        raise FileNotFoundError(f"No question set at {path}")

    rows = yaml.safe_load(path.read_text()) or []
    questions = [EvalQuestion(**row) for row in rows]

    counts = Counter(q.id for q in questions)
    duplicates = sorted(qid for qid, n in counts.items() if n > 1)
    if duplicates:
        raise ValueError(f"duplicate question ids in {path}: {duplicates}")

    return questions


def labelled(questions: list[EvalQuestion]) -> list[EvalQuestion]:
    """Only the questions that can actually be scored.

    The eval is meant to be runnable while the gold set is still being filled in, so
    unlabelled questions are skipped rather than fatal — but the CLI names every one
    it skipped, so a half-labelled set never quietly reports as a full one.
    """
    return [q for q in questions if q.is_labelled]


def unlabelled(questions: list[EvalQuestion]) -> list[EvalQuestion]:
    """The complement of `labelled` — the remaining labelling backlog."""
    return [q for q in questions if not q.is_labelled]
