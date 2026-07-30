# Vision

What this project is, who it's for, and how to tell if we're still building the right thing. See [DECISIONS.md](DECISIONS.md) for architecture-level tradeoffs.

## What this is

**Munich Intel v2: an LLM-extraction pipeline that turns scraped, unstructured Munich AI/deep-tech text into a structured entity-relationship graph — proven by using it to test one real, falsifiable claim about the scene, and checked against a small eval question bank instead of vibes.**

It is **not**:
- A live monitor / scheduler (static, manually re-run snapshots instead).
- A startup-success predictor.
- A data-journalism report where the story is the whole point.

The headline skill on display is the pipeline: raw scraped pages → an LLM turns them into structured entities/relationships → a graph → answers to questions a plain RAG chatbot (v1) couldn't. The momentum story below is *proof the pipeline works*, not the goal itself.

## Why (audience and purpose)

This eventually surfaces on LinkedIn / a resume, so it has to earn its own audience, not just exist.

- **Audience:** Munich founders, investors, and ecosystem people — including the founders already reachable for job conversations — not a generic ML/data audience. Test: would a founder you know actually want to reshare or argue with this?
- **Purpose:** demonstrate AI-engineering skill (structured extraction from unstructured text via LLMs, schema design, graph construction) to people who might actually hire — not analyst or VC-adjacent credibility for its own sake.
- **Personal differentiator:** a mathematician's read on network structure and momentum, not another aggregate funding-stats post.

### Why not just repeat the existing aggregate reports

Munich Startup, Dealroom, and German Startup Monitor already publish aggregate numbers (total funding, round counts, sector share, an employment-impact report). Repeating those would look like restating a press release. What none of them do is link a *specific* company's funding event to its *own* subsequent hiring, compare that link across AI/deep-tech vs. the rest of Munich, or check whether activity survives a news spike. That relational, company-level view is the actual gap — and it's what this pipeline is built to produce.

## Relationship to Munich Intel v1

v1 was a RAG chatbot over unstructured scraped text — a common portfolio project. v2 doesn't replace it; it adds an extraction layer on the same scraped universe, turning chunks into structured entities (`Company`, `FundingRound`, `Investor`, `JobPosting`, `NewsMention`) instead of just retrieval passages. This is a rarer, more in-demand skill than chat-over-docs, and it feeds the existing roadmap's "RAG improvements" and "observability" phases (graph-augmented retrieval, evaluable pipelines).

## Scope, decided

- Static, manually re-runnable snapshot — no scheduler, no live monitoring.
- Entity-relationship schema (`Company`, `FundingRound`, `Investor`, `JobPosting`, `NewsMention`), graph-ready from the start — not flat fields.
- `networkx`, in-memory — not a graph database server.
- Append snapshots over time rather than overwrite, so trend/momentum questions are answerable later.
- Target eventually 100–500 companies, but see build order below — don't scale before the pipeline is proven.

## Scope, still open

| Question | Leaning | Why it's still open |
|---|---|---|
| Munich-only vs. Germany-wide | Munich-only | Tighter, more finishable, matches the founders-you-already-know audience |
| Person/founder entities | Defer | Not needed for the momentum story; the one entity type with real LinkedIn-scraping risk |
| Category as its own graph node vs. a `Company` attribute | Attribute for now | Only worth a node if a question needs to traverse *between* categories — none currently do |
| Discovery source (free directories vs. paid API) | Undecided | Doesn't matter until past the ~20–30 company pilot |
| Human-review gate for new companies | Undecided | Revisit once discovery source is picked |

## The story: is Munich AI/deep-tech momentum real, or just loud?

A 4-beat arc, each beat building on the last — this is the eval set, not a bag of independent questions:

1. **Talk vs. substance** — Do AI/deep-tech companies get more news coverage per euro raised than other Munich startups?
2. **Does the money turn into hiring?** — After a funding round, do job postings actually rise, and how soon?
3. **Is AI different, or is this just how Munich startups behave?** — Run 1–2 for AI/deep-tech vs. the rest of the dataset. This comparison is the actual payoff — without it, it's just stats; with it, it's a real finding either way.
4. **Does it last, or fade?** — After a news spike, does hiring/funding activity continue, or drop once the coverage does?

Parked deliberately, not forgotten: investor co-investment cliques, degree distributions, clustering coefficients. Real network-science questions, but they serve a different story (network structure, for a more technical audience) and would dilute this one if blended in.

## Build order

1. Pick ~20–30 seed companies — small enough to hand-check every extraction for correctness.
2. Define the entity schema in code (pydantic models for `Company`, `FundingRound`, `Investor`, `JobPosting`, `NewsMention`) — the contract extraction has to fill in.
3. Build the extraction step: scraped text → LLM call → structured entities/relations. This is the new engineering piece v1 didn't have.
4. Build the graph from extracted entities with `networkx`.
5. Add the snapshot mechanism — append a timestamped run, don't overwrite.
6. Turn the 4-beat arc into an eval harness: run each question against the graph, check it's answerable and correct.
7. Only once 1–6 work end-to-end on the small set, decide whether to scale up company count.
8. Then build the momentum analysis and the graph visualization for the actual post — an application on top of a working pipeline, not a parallel track.
