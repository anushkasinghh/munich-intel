from functools import lru_cache

import ollama
from groq import Groq

from munich_intel.config import settings

_SYSTEM_PROMPT = """\
You are a Munich startup intelligence assistant. Your job is to answer questions \
about Munich-based startups using only the context provided below.

Rules:
- Answer only from the provided context. If the context does not contain enough \
information to answer, say so clearly.
- Cite the company name when you use information from a context section, \
e.g. "According to [1] NavVis, ..." or "Lilium [3] focuses on...".
- Be concise. One to three paragraphs unless the question requires more detail.
- Do not hallucinate funding numbers, team sizes, or product claims not in the context.\
"""


@lru_cache(maxsize=1)
def _groq_client() -> Groq:
    return Groq(api_key=settings.groq_api_key)


def _build_context_block(chunks: list[dict]) -> str:
    sections = []
    for i, chunk in enumerate(chunks, start=1):
        sections.append(
            f"[{i}] {chunk['company_name']} ({chunk['url']})\n{chunk['chunk_text']}"
        )
    return "\n\n---\n\n".join(sections)


def _messages(query: str, context_chunks: list[dict]) -> list[dict]:
    context_block = _build_context_block(context_chunks)
    user_message = f"Context:\n\n{context_block}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def generate(query: str, context_chunks: list[dict]) -> str:
    msgs = _messages(query, context_chunks)
    if settings.llm_provider == "groq":
        resp = _groq_client().chat.completions.create(model=settings.groq_model, messages=msgs)
        return resp.choices[0].message.content
    resp = ollama.chat(model=settings.ollama_model, messages=msgs, stream=False)
    return resp["message"]["content"]


def generate_stream(query: str, context_chunks: list[dict]):
    msgs = _messages(query, context_chunks)
    if settings.llm_provider == "groq":
        stream = _groq_client().chat.completions.create(
            model=settings.groq_model, messages=msgs, stream=True
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
        return
    for chunk in ollama.chat(model=settings.ollama_model, messages=msgs, stream=True):
        yield chunk["message"]["content"]
