"""CLI entrypoint: python scripts/eval_retrieval.py [--questions PATH] [--ids a,b,c]

Scores RETRIEVAL — which pages come back for a question, and in what order —
against the hand-labelled question set in data/eval/questions.yaml. Separate from
scripts/eval_jobs.py, which scores entity extraction; the two share only
`normalize_url`.

No LLM calls: this embeds each question and queries Qdrant, nothing more. It does
need the Qdrant collection and the local bge-m3 weights, so it is not offline in the
strict sense, but it costs no tokens and is safe to run in a loop while tuning.

Every number is computed from the SAME retrieved list per question (fetched once at
the largest k), so recall@2 and recall@10 can never disagree about what came back.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from munich_intel.config import settings
from munich_intel.embedder import load_model
from munich_intel.eval.qa_dataset import (
    QUESTIONS_PATH,
    EvalQuestion,
    labelled,
    load_questions,
    unlabelled,
)
from munich_intel.eval.retrieval_metrics import (
    company_diversity,
    mrr,
    pooled_recall,
    recall_counts_at_k,
)
from munich_intel.retriever import retrieve

console = Console()

# Reported at every k so the cost of the current cutoff is visible next to what a
# larger one would buy. k=2 is today's setting; k=10 is roughly where recall stops
# improving for a 21-company corpus.
K_VALUES = (2, 5, 10)
FETCH_K = max(K_VALUES)

KIND_ORDER = ("single-company", "multi-company", "comparison", "aggregation")


def build_client() -> QdrantClient:
    """Same connection logic as the rest of the app: qdrant_url wins if set."""
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def score(question: EvalQuestion, hits: list[dict]) -> dict:
    """Every metric for one question, from one retrieved list."""
    urls = [hit["url"] for hit in hits]
    return {
        "question": question,
        "recall": {k: recall_counts_at_k(urls, question.relevant, k) for k in K_VALUES},
        "mrr": mrr(urls, question.relevant),
        "diversity": {k: company_diversity(hits, k) for k in K_VALUES},
    }


def _add_row(table: Table, label: str, rows: list[dict], bold: bool = False) -> None:
    """A pooled row: recall recomputed from summed counts, MRR and diversity averaged.

    Recall pools because it is a ratio of countable things — pages found over pages
    that existed — and averaging its per-question rates would let a question with one
    relevant page outweigh one with eight.

    MRR and diversity are means instead, because there is nothing to pool: both are
    already per-question facts about one result list ("how far down was the first good
    answer", "how many companies did the reader see"), and every question's reader
    counts once. Different operations on purpose, not an inconsistency.
    """
    cells = [label]
    for k in K_VALUES:
        pooled = pooled_recall([row["recall"][k] for row in rows])
        cells.append(f"{pooled.recall:.2f}" if rows else "—")
    cells.append(f"{sum(r['mrr'] for r in rows) / len(rows):.2f}" if rows else "—")
    for k in K_VALUES:
        cells.append(f"{sum(r['diversity'][k] for r in rows) / len(rows):.1f}" if rows else "—")
    cells.append(str(len(rows)))
    if bold:
        cells = [f"[bold]{cell}[/bold]" for cell in cells]
    table.add_row(*cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--ids", help="comma-separated question ids (default: all labelled)")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    todo = labelled(questions)
    if args.ids:
        wanted = set(args.ids.split(","))
        todo = [q for q in todo if q.id in wanted]

    backlog = unlabelled(questions)
    if not todo:
        console.print(
            f"[yellow]No labelled questions in {args.questions}.[/yellow] "
            f"{len(backlog)} awaiting labels — fill in relevant_urls to score them."
        )
        _print_backlog(backlog)
        return

    model = load_model()
    client = build_client()

    results = [
        score(question, retrieve(question.question, model, client, settings.collection_name, FETCH_K))
        for question in todo
    ]

    per_question = Table(
        title=f"Retrieval vs. gold (top_k in use: {settings.retrieval_top_k}, "
        f"chunk {settings.chunk_size}w/{settings.chunk_overlap})  "
        f"[dim]† = aggregation, expected to fail[/dim]"
    )
    # The id column folds rather than truncating: in a narrow terminal rich shrinks
    # whatever can wrap, and a wrapped id is readable where an ellipsed score is not.
    per_question.add_column("Question", style="cyan", overflow="fold")
    per_question.add_column("Kind", style="dim", overflow="fold")
    per_question.add_column("Rel", justify="right")
    for k in K_VALUES:
        per_question.add_column(f"R@{k}", justify="right")
    per_question.add_column("MRR", justify="right")
    for k in K_VALUES:
        per_question.add_column(f"C@{k}", justify="right")

    for row in sorted(results, key=lambda r: (KIND_ORDER.index(r["question"].kind), r["question"].id)):
        question = row["question"]
        # Expected-to-fail questions are marked in place so a low score is never
        # mistaken for a regression introduced by a retrieval change. A "†" rather
        # than a word, to keep the id column narrow enough to read at a glance.
        label = f"{question.id} [dim]†[/dim]" if question.expected_to_fail else question.id
        per_question.add_row(
            label,
            question.kind,
            str(row["recall"][K_VALUES[0]].total),
            *[f"{row['recall'][k].recall:.2f}" for k in K_VALUES],
            f"{row['mrr']:.2f}",
            *[str(row["diversity"][k]) for k in K_VALUES],
        )
    console.print(per_question)

    pooled = Table(title="Pooled by kind (recall from summed counts; MRR and diversity are means)")
    pooled.add_column("Kind", style="cyan")
    for k in K_VALUES:
        pooled.add_column(f"R@{k}", justify="right")
    pooled.add_column("MRR", justify="right")
    for k in K_VALUES:
        pooled.add_column(f"C@{k}", justify="right")
    pooled.add_column("Qs", justify="right")

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_kind[row["question"].kind].append(row)
    for kind in KIND_ORDER:
        if by_kind[kind]:
            _add_row(pooled, kind, by_kind[kind])

    # The headline excludes aggregation questions: they are unanswerable by
    # retrieval at any k, so folding them in would make every future retrieval
    # change look smaller than it is. They are shown on their own row above and
    # in the "all" row below, which is the honest total.
    scorable = [row for row in results if not row["question"].expected_to_fail]
    pooled.add_section()
    _add_row(pooled, "retrievable (excl. aggregation)", scorable, bold=True)
    _add_row(pooled, "all questions", results)
    console.print(pooled)

    _print_backlog(backlog)


def _print_backlog(backlog: list[EvalQuestion]) -> None:
    if not backlog:
        return
    table = Table(title="Unlabelled — skipped, not scored as zero")
    table.add_column("Question id", style="cyan")
    table.add_column("Kind", style="dim")
    table.add_column("Question", overflow="fold")
    for question in backlog:
        table.add_row(question.id, question.kind, question.question)
    console.print(table)


if __name__ == "__main__":
    main()
