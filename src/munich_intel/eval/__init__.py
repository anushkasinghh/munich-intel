"""Eval harness (EVAL_DESIGN.md).

Thin by design: jobs only, 5 companies, no LLM calls, no network, no judge.
The point is to put a number on `extractor.extract_jobs` before changing it, so
the ATS-aware scraping fix can be measured as a delta rather than asserted.

`metrics` is pure (zero I/O). `datasets` is the only module that touches disk.
"""
