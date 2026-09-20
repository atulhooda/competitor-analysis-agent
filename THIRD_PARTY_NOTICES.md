# Third-party notices

This project adapts small parts of the following MIT-licensed projects. Their copyright
and permission notices are reproduced below, as the MIT License requires.

## Str1nX03/Competitor-Research

<https://github.com/Str1nX03/Competitor-Research> (MIT)

Adapted patterns:
- the pydantic-settings `Settings` class with a cached `get_settings()` (`app/config.py`)
- the `get_llm()` factory shape (`app/llm/factory.py`)

```
MIT License

Copyright (c) 2026 Dravin Kumar Sharma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## gokborayilmaz/competitor-analysis-agent

<https://github.com/gokborayilmaz/competitor-analysis-agent> (MIT)

Adapted:
- the `CompetitorProfile` schema from `shemas.py` (`app/domain/competitor_profile.py`), extended
  with evidence citations, structured pricing tiers and deterministic content statistics;
- the pattern of studying each competitor in isolation and synthesizing only from structured
  evidence (`app/services/profiles.py`, `app/prompts/competitor_profile.py`; the prompt text is new).

```
MIT License

Copyright (c) 2024 Upsonic Teknoloji A.Ş.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Repository `ShreyashSoni/agentic_blog_generator` has no license, so no code or text from
it is used. See `MIGRATION_PLAN.md` §3.
