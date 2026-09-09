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
    negative_controls,
    unlabelled,
)
from munich_intel.eval.retrieval_metrics import (
    company_diversity,
    mrr,
    pooled_recall,
    recall_counts_at_k,
    threshold_gap,
    top_score,
)
from munich_intel.retriever import retrieve

console = Console()

# Reported at every k so the cost of the current cutoff is visible next to what a
# larger one would buy. k=2 is today's setting; k=10 is roughly where recall stops
# improving for a 21-company corpus.
K_VALUES = (2, 5, 10)
FETCH_K = max(K_VALUES)

KIND_ORDER = ("single-company", "multi-company", "comparison", "aggregation")

BLANK = "—"

_CAP_LABEL = "off"


def build_client() -> QdrantClient:
    """Same connection logic as the rest of the app: qdrant_url wins if set."""
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def score(question: EvalQuestion, hits: list[dict]) -> dict:
    """Every applicable metric for one question, from one retrieved list.

    Metrics that do not apply are None rather than 0.0, so the tables can print a
    dash. A negative control scoring 0.0 recall is not the same fact as a real
    question scoring 0.0, and collapsing them would make the pooled rows lie.
    """
    urls = [hit["url"] for hit in hits]
    scores_recall = question.scores_recall
    return {
        "question": question,
        "recall": (
            {k: recall_counts_at_k(urls, question.relevant, k) for k in K_VALUES}
            if scores_recall
            else None
        ),
        "mrr": mrr(urls, question.relevant) if scores_recall else None,
        "diversity": (
            {k: company_diversity(hits, k) for k in K_VALUES}
            if question.scores_diversity
            else None
        ),
        "top_score": top_score(hits),
    }


def _pooled_cells(rows: list[dict]) -> list[str]:
    """The numeric cells of a pooled row.

    Recall pools because it is a ratio of countable things — pages found over pages
    that existed — and averaging its per-question rates would let a question with one
    relevant page outweigh one with eight.

    MRR and diversity are means instead, because there is nothing to pool: both are
    already per-question facts about one result list ("how far down was the first good
    answer", "how many companies did the reader see"), and every question's reader
    counts once. Different operations on purpose, not an inconsistency.

    Each metric averages over only the questions it applies to, so a single-company
    question never dilutes a diversity figure and a negative control never dilutes
    recall.
    """
    scored = [r for r in rows if r["recall"] is not None]
    diverse = [r for r in rows if r["diversity"] is not None]

    cells = []
    for k in K_VALUES:
        pooled = pooled_recall([r["recall"][k] for r in scored])
        cells.append(f"{pooled.recall:.2f}" if scored else BLANK)
    cells.append(f"{sum(r['mrr'] for r in scored) / len(scored):.2f}" if scored else BLANK)
    for k in K_VALUES:
        cells.append(
            f"{sum(r['diversity'][k] for r in diverse) / len(diverse):.1f}" if diverse else BLANK
        )
    return cells


def _add_pooled_row(table: Table, label: str, rows: list[dict], bold: bool = False) -> None:
    cells = [label, *_pooled_cells(rows), str(len(rows))]
    if bold:
        cells = [f"[bold]{cell}[/bold]" for cell in cells]
    table.add_row(*cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--ids", help="comma-separated question ids (default: all labelled)")
    parser.add_argument(
        "--per-company",
        type=int,
        default=2,
        help="max chunks kept from any one company (default: 2). Use 0 to disable "
        "capping and reproduce the pre-3b baseline.",
    )
    parser.add_argument(
        "--cap-key",
        default="url",
        choices=["url", "company_slug"],
        help="what the cap counts: 'url' (default, one page) or 'company_slug'. "
        "Measured: per-company capping raises diversity but LOSES recall, because "
        "recall counts distinct pages and most answers live in 2-3 pages of one "
        "company. See EVAL_DESIGN.md 3b.",
    )
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

    # Capping is prefix-stable: the greedy first pass fills in score order, so the
    # top 2 of a capped top-10 are the same two the app would get asking for k=2.
    # That is what lets one fetch at FETCH_K serve recall@2, @5 and @10 alike.
    per_company = args.per_company or None
    global _CAP_LABEL
    _CAP_LABEL = f"{per_company}/{args.cap_key}" if per_company else "off"
    results = [
        score(
            q,
            retrieve(
                q.question, model, client, settings.collection_name, FETCH_K,
                per_company=per_company, cap_key=args.cap_key,
            ),
        )
        for q in todo
    ]

    _print_per_question(results)
    _print_pooled(results)
    _print_controls(results)
    _print_backlog(backlog)


def _print_per_question(results: list[dict]) -> None:
    table = Table(
        title=f"Retrieval vs. gold (top_k in use: {settings.retrieval_top_k}, "
        f"chunk {settings.chunk_size}w/{settings.chunk_overlap}, "
        f"cap {_CAP_LABEL})  "
        f"[dim]† aggregation, expected to fail · ‡ negative control[/dim]"
    )
    # The id column folds rather than truncating: in a narrow terminal rich shrinks
    # whatever can wrap, and a wrapped id is readable where an ellipsed score is not.
    table.add_column("Question", style="cyan", overflow="fold")
    table.add_column("Kind", style="dim", overflow="fold")
    table.add_column("Rel", justify="right")
    for k in K_VALUES:
        table.add_column(f"R@{k}", justify="right")
    table.add_column("MRR", justify="right")
    for k in K_VALUES:
        table.add_column(f"C@{k}", justify="right")
    table.add_column("Top", justify="right")

    for row in sorted(results, key=lambda r: (KIND_ORDER.index(r["question"].kind), r["question"].id)):
        question = row["question"]
        # Marked in place so a low score is never mistaken for a regression
        # introduced by a retrieval change.
        mark = " [dim]‡[/dim]" if question.expects_empty else (" [dim]†[/dim]" if question.expected_to_fail else "")
        recall, diversity = row["recall"], row["diversity"]
        table.add_row(
            f"{question.id}{mark}",
            question.kind,
            str(recall[K_VALUES[0]].total) if recall else BLANK,
            *[f"{recall[k].recall:.2f}" if recall else BLANK for k in K_VALUES],
            f"{row['mrr']:.2f}" if row["mrr"] is not None else BLANK,
            *[str(diversity[k]) if diversity else BLANK for k in K_VALUES],
            f"{row['top_score']:.2f}",
        )
    console.print(table)


def _print_pooled(results: list[dict]) -> None:
    table = Table(
        title="Pooled by kind (recall from summed counts; MRR and diversity are means "
        "over the questions each applies to)"
    )
    table.add_column("Kind", style="cyan")
    for k in K_VALUES:
        table.add_column(f"R@{k}", justify="right")
    table.add_column("MRR", justify="right")
    for k in K_VALUES:
        table.add_column(f"C@{k}", justify="right")
    table.add_column("Qs", justify="right")

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_kind[row["question"].kind].append(row)
    for kind in KIND_ORDER:
        if by_kind[kind]:
            _add_pooled_row(table, kind, by_kind[kind])

    # The headline excludes aggregation questions and negative controls: neither is
    # answerable by retrieval at any k, so folding them in would make every future
    # retrieval change look smaller than it is. Both appear on their own rows above
    # and in the "all" row below, which is the honest total.
    headline = [
        r for r in results if not r["question"].expected_to_fail and not r["question"].expects_empty
    ]
    table.add_section()
    _add_pooled_row(table, "retrievable (headline)", headline, bold=True)
    _add_pooled_row(table, "all questions", results)
    console.print(table)


def _print_controls(results: list[dict]) -> None:
    """Whether an unanswerable question can be told apart from an answerable one.

    Reported separately because it is not a recall question at all: it asks whether
    any score cutoff could exist. Without it the eval would have no way to notice a
    change that improves recall by returning more of everything, junk included.
    """
    controls = [r for r in results if r["question"].expects_empty]
    if not controls:
        return

    answerable = [r["top_score"] for r in results if r["recall"] is not None]
    unanswerable = [r["top_score"] for r in controls]
    gap = threshold_gap(answerable, unanswerable)

    table = Table(title="Negative controls — can an unanswerable question be detected?")
    table.add_column("Measure", style="cyan")
    table.add_column("Score", justify="right")
    table.add_row("worst top-score on an answerable question", f"{min(answerable):.3f}" if answerable else BLANK)
    table.add_row("best top-score on a negative control", f"{max(unanswerable):.3f}")
    table.add_section()
    # Three states, not two: "no gap demonstrated" is a different claim from "no gap
    # exists", and with nothing labelled to compare against only the first is honest.
    if not answerable:
        verdict = "[yellow]not measurable — no labelled answerable questions yet[/yellow]"
    elif gap > 0:
        verdict = f"[green]{gap:+.3f} — a cutoff exists in this range[/green]"
    else:
        verdict = f"[red]{gap:+.3f} — no cutoff can separate them[/red]"
    table.add_row("[bold]threshold gap[/bold]", verdict)
    console.print(table)


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
