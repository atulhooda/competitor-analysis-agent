"""Versioned prompts. Each module defines ``VERSION``, ``SYSTEM``, a ``render()`` function
and the structured-output schema the model must return.

``VERSION`` is stored on every row a prompt produces (analyses, summaries, profiles,
reports) and on every LLM call, so outputs stay traceable to the exact prompt. Change a
prompt's wording or schema → bump its version.

Competitor pages are untrusted input: prompts wrap them in delimiters and tell the model
to treat them as data. The model has no tools, so injected text can at worst skew one
analysis, which schema validation and grounding checks then constrain.
"""
