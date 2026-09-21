"""The import file format: one Markdown file with YAML frontmatter, written by a person,
parsed into exactly the ``ArticleContent`` the writing step produces. No LLM, no database.

    ---
    title: ...            the article's title (the site renders it; no H1 in the body)
    description: ...      the meta description
    primary_keyword: ...  what the piece is about
    target_audience: ...  who it is for
    content_type: guide   guide | listicle | comparison | research | tutorial | article
    tags: [3 to 6]        the site's post tags, lowercased when published
    slug: ...             optional; derived from the title otherwise
    sources:              the pages the body cites, one per [S…] marker it uses
      - label: S1
        title: ...
        url: https://...
    ---

    The body, in Markdown: paragraphs before the first `##` are the introduction, every
    `##` starts a section, `###` is a subheading, `-`/`1.` are lists, and `[S1]` marks a
    claim's source, exactly as the writing step marks one.

The body is **prose**, not markup: everything MDX could read as something else is escaped
when the post is rendered, so a file that contains a tag, a brace or an ESM line is refused
here rather than silently published as literal text (see ``app.cms.github.mdx.validate``,
which checks the composed file again before anything is written).
"""

import re
from typing import Any

from pydantic import BaseModel, Field

from app.cms.github.mdx import site_slug, split_frontmatter
from app.core.errors import PermanentError
from app.domain.analysis import ContentFormat
from app.domain.articles import (
    ArticleContent,
    BlockType,
    ContentBlock,
    ContentSection,
    SectionKind,
)
from app.services.article_content import (
    citations,
    clean_citations,
    slugify,
    structural_problems,
    word_count,
)
from app.services.article_render import safe_url

IMPORT_VERSION = "import/1"
# The formats a person may choose; the others (opinion, interview, case_study) are the
# competitor engine's reading of someone else's page, never a choice made here.
ALLOWED_FORMATS: tuple[ContentFormat, ...] = (ContentFormat.GUIDE, ContentFormat.LISTICLE, ContentFormat.COMPARISON, ContentFormat.RESEARCH, ContentFormat.TUTORIAL, ContentFormat.ARTICLE)  # fmt: skip
MIN_TAGS, MAX_TAGS, MAX_TAG_LENGTH = 3, 6, 40
REQUIRED_FIELDS = ("title", "description", "primary_keyword", "target_audience", "content_type", "tags", "sources")  # fmt: skip
_LABEL = re.compile(r"^S\d+$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")
# What MDX would read as something other than prose (see app/cms/github/mdx.py).
_ESM_LINE = re.compile(r"^(import|export)\s", re.MULTILINE)
_JSX_OPEN = re.compile(r"<(?=[A-Za-z/])")


class ArticleFileError(PermanentError):
    """The file isn't a publishable article: every problem is named."""

    def __init__(self, problems: list[str], *, path: str | None = None) -> None:
        where = f"{path}: " if path else ""
        super().__init__(where + "; ".join(problems))
        self.problems = problems


class ImportedSource(BaseModel):
    """One page the body cites, named by the marker the body uses for it."""

    label: str  # S1, S2, ...
    title: str
    url: str


class ImportedArticle(BaseModel):
    """A parsed file: the same shape the writing step hands to validation and publishing."""

    title: str
    description: str
    primary_keyword: str
    target_audience: str
    content_type: ContentFormat
    tags: list[str]
    slug: str
    sources: list[ImportedSource]
    content: ArticleContent
    word_count: int
    cited_labels: list[str] = Field(description="The source labels the body actually cites")

    @property
    def headings(self) -> list[str]:
        return [s.heading for s in self.content.sections if s.heading]


# ── the body ─────────────────────────────────────────────────────────────────


class _Body:
    """Markdown lines → sections, in one pass. Paragraphs are joined across wrapped lines;
    a list item continues on an indented or plain following line."""

    def __init__(self) -> None:
        self.sections: list[ContentSection] = []
        self.blocks: list[ContentBlock] = []
        self.kind = SectionKind.INTRODUCTION
        self.heading: str | None = None
        self.paragraph: list[str] = []
        self.items: list[str] = []
        self.ordered = False

    def flush_block(self) -> None:
        if self.paragraph:
            self.blocks.append(ContentBlock(type=BlockType.PARAGRAPH, text=" ".join(self.paragraph)))  # fmt: skip
            self.paragraph = []
        if self.items:
            self.blocks.append(ContentBlock(type=BlockType.LIST, items=list(self.items), ordered=self.ordered))  # fmt: skip
            self.items = []

    def flush_section(self) -> None:
        self.flush_block()
        if self.blocks:
            self.sections.append(ContentSection(kind=self.kind, heading=self.heading, blocks=self.blocks))  # fmt: skip
        self.blocks = []

    def start_section(self, heading: str) -> None:
        self.flush_section()
        self.kind, self.heading = SectionKind.BODY, heading

    def subheading(self, text: str) -> None:
        self.flush_block()
        self.blocks.append(ContentBlock(type=BlockType.SUBHEADING, text=text))

    def item(self, text: str, *, ordered: bool) -> None:
        if self.items and ordered is not self.ordered:
            self.flush_block()
        self.paragraph = []
        self.ordered = ordered
        self.items.append(text)

    def line(self, text: str) -> None:
        if self.items:  # a wrapped list item continues
            self.items[-1] += " " + text
        else:
            self.paragraph.append(text)

    def done(self) -> list[ContentSection]:
        self.flush_section()
        return self.sections


def content_from_markdown(title: str, description: str, body: str) -> ArticleContent:
    """The body as article content: paragraphs before the first ``##`` are the introduction,
    every ``##`` opens a body section, ``###`` is a subheading, ``-`` and ``1.`` are lists."""
    out = _Body()
    for raw in body.splitlines():
        line = raw.rstrip()
        if not line.strip():
            out.flush_block()
            continue
        heading = _HEADING.match(line.strip())
        if heading is not None:
            level, text = len(heading.group(1)), " ".join(heading.group(2).split())
            if level == 1:
                raise ArticleFileError(["the body has an H1 (`# …`): the title comes from the frontmatter and the site renders it; use `##` for sections"])  # fmt: skip
            if level == 2:
                out.start_section(text)
            else:
                out.subheading(text)
            continue
        bullet = _BULLET.match(line)
        numbered = _NUMBERED.match(line)
        if bullet is not None:
            out.item(bullet.group(1).strip(), ordered=False)
        elif numbered is not None:
            out.item(numbered.group(1).strip(), ordered=True)
        else:
            out.line(line.strip())
    return ArticleContent(title=title, description=description, sections=out.done())


def markup_problems(body: str) -> list[str]:
    """What a hand-written body may not contain: MDX would read it as code, not prose."""
    problems = []
    if _ESM_LINE.search(body):
        problems.append("the body has a line starting with `import` or `export`: MDX would read it as code")  # fmt: skip
    if _JSX_OPEN.search(body):
        problems.append("the body has a `<` before a letter (an HTML or MDX tag): the body is prose, not markup")  # fmt: skip
    if "{" in body or "}" in body:
        problems.append("the body has a `{` or `}`: MDX would evaluate it as an expression")
    return problems


# ── the file ─────────────────────────────────────────────────────────────────


def _text(fields: dict[str, Any], name: str, problems: list[str], *, limit: int = 500) -> str:
    value = fields.get(name)
    if not isinstance(value, str) or not value.strip():
        problems.append(f"frontmatter: {name} is required and must be a non-empty string")
        return ""
    text = " ".join(value.split())
    if len(text) > limit:
        problems.append(f"frontmatter: {name} is longer than {limit} characters")
    return text


def _tags(fields: dict[str, Any], problems: list[str]) -> list[str]:
    raw = fields.get("tags")
    if not isinstance(raw, list) or not all(isinstance(t, str) for t in raw):
        problems.append("frontmatter: tags must be a list of strings")
        return []
    tags: list[str] = []
    for tag in raw:
        clean = " ".join(str(tag).split())[:MAX_TAG_LENGTH].strip(" -")
        if clean and clean not in tags:
            tags.append(clean)
    if not MIN_TAGS <= len(tags) <= MAX_TAGS:
        problems.append(f"frontmatter: {len(tags)} distinct tag(s); the site wants {MIN_TAGS}-{MAX_TAGS}")  # fmt: skip
    return tags


def _format(fields: dict[str, Any], problems: list[str]) -> ContentFormat:
    value = str(fields.get("content_type") or "").strip().lower()
    chosen = next((f for f in ALLOWED_FORMATS if f.value == value), None)
    if chosen is None:
        problems.append(f"frontmatter: content_type {value or '(missing)'!r} isn't one of " + ", ".join(f.value for f in ALLOWED_FORMATS))  # fmt: skip
    return chosen or ContentFormat.GUIDE


def _sources(fields: dict[str, Any], problems: list[str]) -> list[ImportedSource]:
    raw = fields.get("sources")
    if not isinstance(raw, list) or not raw:
        problems.append("frontmatter: sources must list at least one {label, title, url}: an article states where its facts come from")  # fmt: skip
        return []
    sources: list[ImportedSource] = []
    labels: set[str] = set()
    urls: set[str] = set()
    for index, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            problems.append(f"frontmatter: source {index} must be a mapping of label, title and url")  # fmt: skip
            continue
        label = str(entry.get("label") or "").strip()
        title = " ".join(str(entry.get("title") or "").split())
        url = safe_url(str(entry.get("url") or ""))
        if not _LABEL.match(label):
            problems.append(f"frontmatter: source {index} has label {label or '(missing)'!r}; labels are S1, S2, … (the markers the body uses)")  # fmt: skip
            continue
        if label in labels:
            problems.append(f"frontmatter: source label {label} is used twice")
            continue
        if url is None:
            problems.append(f"frontmatter: source {label} needs an absolute http(s) url")
            continue
        if url in urls:
            problems.append(f"frontmatter: source {label} repeats the url {url}")
            continue
        if not title:
            problems.append(f"frontmatter: source {label} needs a title (it is shown as the link text)")  # fmt: skip
            continue
        labels.add(label)
        urls.add(url)
        sources.append(ImportedSource(label=label, title=title[:500], url=url))
    return sources


def parse_article_file(text: str, *, min_words: int, path: str | None = None) -> ImportedArticle:
    """One file → an article, or ``ArticleFileError`` naming every problem at once."""
    fields, body = split_frontmatter(text)
    if fields is None:
        raise ArticleFileError(["the file must start with YAML frontmatter between `---` lines (" + ", ".join(REQUIRED_FIELDS) + ")"], path=path)  # fmt: skip
    problems: list[str] = []
    title = _text(fields, "title", problems, limit=200)
    description = _text(fields, "description", problems, limit=320)
    keyword = _text(fields, "primary_keyword", problems, limit=120)
    audience = _text(fields, "target_audience", problems, limit=200)
    content_type = _format(fields, problems)
    tags = _tags(fields, problems)
    sources = _sources(fields, problems)
    problems += markup_problems(body)
    if problems:
        raise ArticleFileError(problems, path=path)
    content = content_from_markdown(title, description, body)
    labels = {s.label for s in sources}
    cited = [label for c in citations(content) for label in c.labels]
    missing = sorted({label for label in cited if label not in labels}, key=lambda label: int(label[1:]))  # fmt: skip
    if missing:
        problems.append(f"the body cites {', '.join(missing)}, which no source in the frontmatter defines: add the source, or remove the marker")  # fmt: skip
    unused = sorted(labels - set(cited), key=lambda label: int(label[1:]))
    if unused:
        problems.append(f"source(s) {', '.join(unused)} are listed but never cited in the body: cite them, or remove them")  # fmt: skip
    # ``labels=cited``: unknown markers are reported once, above, in this file's own words.
    problems += structural_problems(content, min_words=min_words, labels=set(cited))
    slug = site_slug(str(fields.get("slug") or "").strip() or slugify(title))
    if not slug:
        problems.append(f"no usable slug from {fields.get('slug') or title!r}: the site's slugs are lowercase ASCII letters, digits and hyphens")  # fmt: skip
    if problems:
        raise ArticleFileError(problems, path=path)
    content, _ = clean_citations(content, labels)  # canonical markers; nothing is removed
    return ImportedArticle(
        title=title, description=description, primary_keyword=keyword, target_audience=audience,
        content_type=content_type, tags=tags, slug=slug, sources=sources, content=content,
        word_count=word_count(content), cited_labels=list(dict.fromkeys(cited)),
    )  # fmt: skip


__all__ = [
    "ALLOWED_FORMATS",
    "IMPORT_VERSION",
    "ArticleFileError",
    "ImportedArticle",
    "ImportedSource",
    "content_from_markdown",
    "markup_problems",
    "parse_article_file",
]
