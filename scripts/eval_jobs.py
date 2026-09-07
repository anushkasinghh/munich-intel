"""CLI entrypoint: python scripts/eval_jobs.py [--slugs a,b,c]

Scores extractor.extract_jobs against the hand-labelled gold set (EVAL_DESIGN.md).
Reads only data/eval/gold_jobs/ and data/entities/ — no LLM calls, no network, so it
is safe to run in a loop while tuning the extractor.

Prints every false positive and false negative URL: that list is the raw material
for data/eval/error_tags.yaml, and the ranked tag counts at the bottom are the
extraction backlog.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from munich_intel.eval.datasets import GOLD_DIR, load_error_tags, load_gold_jobs, load_predicted_jobs
from munich_intel.eval.metrics import (
    duplicate_url_groups,
    field_accuracy,
    field_comparable_count,
    match,
    normalize_url,
    prf1,
    prf1_from_counts,
)

console = Console()

UNTAGGED = "(untagged)"


def discover_slugs() -> list[str]:
    """Every company with a gold file — the eval set is defined by what's labelled."""
    return sorted(path.stem for path in GOLD_DIR.glob("*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slugs", help="comma-separated company slugs (default: all gold files)")
    args = parser.parse_args()

    slugs = args.slugs.split(",") if args.slugs else discover_slugs()
    if not slugs:
        console.print(f"[red]No gold files in {GOLD_DIR}[/red]")
        return

    error_tags = load_error_tags()

    scores = Table(title="Job extraction vs. gold")
    scores.add_column("Company", style="cyan")
    scores.add_column("Gold", justify="right")
    scores.add_column("Pred", justify="right")
    for column in ("TP", "FP", "FN"):
        scores.add_column(column, justify="right")
    for column in ("P", "R", "F1"):
        scores.add_column(column, justify="right")

    total_tp = total_fp = total_fn = 0
    all_matched: list[tuple] = []
    # (kind, slug, url, weight) — weight is how many raw postings that URL key stands
    # for, so a careers page that produced 14 bogus "jobs" ranks above one that
    # produced 1, even though both score as a single FP.
    misses: list[tuple[str, str, str, int]] = []
    collapses: list[tuple[str, str, int]] = []  # (slug, url, count)

    for slug in slugs:
        gold = load_gold_jobs(slug)
        predicted = load_predicted_jobs(slug)

        matched, false_positives, false_negatives = match(predicted, gold)
        result = prf1(
            {normalize_url(str(p.url)) for p in predicted},
            {normalize_url(str(g.url)) for g in gold},
        )

        duplicates = duplicate_url_groups(predicted)
        total_tp += result.tp
        total_fp += result.fp
        total_fn += result.fn
        all_matched.extend(matched)
        misses.extend(
            ("FP", slug, str(p.url), len(duplicates.get(normalize_url(str(p.url)), [p])))
            for p in false_positives
        )
        misses.extend(("FN", slug, str(g.url), 1) for g in false_negatives)
        collapses.extend((slug, url, len(group)) for url, group in duplicates.items())

        scores.add_row(
            slug,
            str(len(gold)),
            str(len(predicted)),
            str(result.tp),
            str(result.fp),
            str(result.fn),
            f"{result.precision:.2f}",
            f"{result.recall:.2f}",
            f"{result.f1:.2f}",
        )

    # Recomputed from pooled counts, never averaged from the per-company rates —
    # averaging would let a 0-posting company outweigh a 14-posting one.
    aggregate = prf1_from_counts(total_tp, total_fp, total_fn)
    scores.add_section()
    scores.add_row(
        "[bold]all (pooled)[/bold]",
        f"[bold]{total_tp + total_fn}[/bold]",
        f"[bold]{total_tp + total_fp}[/bold]",
        f"[bold]{total_tp}[/bold]",
        f"[bold]{total_fp}[/bold]",
        f"[bold]{total_fn}[/bold]",
        f"[bold]{aggregate.precision:.2f}[/bold]",
        f"[bold]{aggregate.recall:.2f}[/bold]",
        f"[bold]{aggregate.f1:.2f}[/bold]",
    )
    console.print(scores)

    fields = Table(title="Field accuracy (correctly-found postings only)")
    fields.add_column("Field", style="cyan")
    fields.add_column("Correct", justify="right")
    fields.add_column("Graded", justify="right")
    fields.add_column("Accuracy", justify="right")
    for field in ("title", "location"):
        graded = field_comparable_count(all_matched, field)
        accuracy = field_accuracy(all_matched, field)
        fields.add_row(field, str(round(accuracy * graded)), str(graded), f"{accuracy:.2f}")
    console.print(fields)

    if collapses:
        collapse_table = Table(
            title="Postings sharing one URL (score as a single FP/TP — see metrics.py)"
        )
        collapse_table.add_column("Company", style="cyan")
        collapse_table.add_column("URL", overflow="fold")
        collapse_table.add_column("Postings", justify="right")
        for slug, url, count in sorted(collapses, key=lambda row: row[2], reverse=True):
            collapse_table.add_row(slug, url, str(count))
        console.print(collapse_table)

    if misses:
        miss_table = Table(title="Every FP and FN (fill these into data/eval/error_tags.yaml)")
        miss_table.add_column("Kind", style="cyan")
        miss_table.add_column("Company", style="cyan")
        miss_table.add_column("URL", overflow="fold")
        miss_table.add_column("Postings", justify="right")
        miss_table.add_column("Tag")
        for kind, slug, url, weight in misses:
            tag = _tag_for(error_tags, url)
            style = "" if tag != UNTAGGED else "dim"
            miss_table.add_row(
                kind, slug, url, str(weight), f"[{style}]{tag}[/{style}]" if style else tag
            )
        console.print(miss_table)

        keys = Counter(_tag_for(error_tags, url) for _, _, url, _ in misses)
        postings = Counter()
        for _, _, url, weight in misses:
            postings[_tag_for(error_tags, url)] += weight

        ranked = Table(title="Error taxonomy — this ranking is the extraction backlog")
        ranked.add_column("Tag", style="cyan")
        ranked.add_column("URL keys", justify="right")
        ranked.add_column("Postings", justify="right")
        for tag, count in postings.most_common():
            ranked.add_row(tag, str(keys[tag]), str(count))
        console.print(ranked)


def _tag_for(error_tags: dict[str, str], url: str) -> str:
    """Tags may be written with or without tracking params — try both spellings."""
    return error_tags.get(url) or error_tags.get(normalize_url(url), UNTAGGED)


if __name__ == "__main__":
    main()
