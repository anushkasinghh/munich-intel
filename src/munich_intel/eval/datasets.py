"""Loading the two sides of the jobs eval: hand-labelled gold, and extractor output.

The only module in `eval/` that touches disk. Both loaders parse through
`entities.JobPosting`, so a gold file that drifts from the schema fails loudly at
load time instead of quietly scoring wrong — the gold set is hand-written, which is
exactly where a typo'd field name would otherwise slip through unnoticed.
"""

import json
from pathlib import Path

import yaml

from munich_intel.entities import JobPosting

GOLD_DIR = Path("data/eval/gold_jobs")
ENTITIES_DIR = Path("data/entities")
ERROR_TAGS_PATH = Path("data/eval/error_tags.yaml")


def _load_postings(path: Path) -> list[JobPosting]:
    """Parse a jobs JSON file, or return [] if it does not exist.

    A missing file is a valid zero, not an error: a company can genuinely have no
    postings, and `augmented-industries` has an empty predicted file for exactly
    that reason (rightly or wrongly — that is what the eval is for).
    """
    if not path.exists():
        return []
    return [JobPosting(**row) for row in json.loads(path.read_text())]


def load_gold_jobs(slug: str, gold_dir: Path = GOLD_DIR) -> list[JobPosting]:
    """The hand-labelled truth for one company: data/eval/gold_jobs/{slug}.json.

    Same schema as the extractor's own output so the two are directly comparable,
    but labelled by reading the scraped page text in data/raw/ — never by editing
    the extractor's output, which would bake its mistakes into the answer key.
    """
    return _load_postings(gold_dir / f"{slug}.json")


def load_predicted_jobs(slug: str, entities_dir: Path = ENTITIES_DIR) -> list[JobPosting]:
    """What the extractor actually produced: data/entities/{slug}_jobs.json."""
    return _load_postings(entities_dir / f"{slug}_jobs.json")


def load_error_tags(path: Path = ERROR_TAGS_PATH) -> dict[str, str]:
    """The hand-filled {url: tag} map explaining *why* each false positive is wrong.

    Precision alone says how often the extractor is wrong; this says what kind of
    wrong, which is what turns a score into a backlog. Tags are invented while
    labelling, not decreed up front — see EVAL_DESIGN.md.
    """
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}
