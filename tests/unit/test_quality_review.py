"""Phase 6 review helpers: how much room a revision is given. A revision returns the whole
article again, so a budget sized for the *target* length cuts long articles off mid-rewrite:
the answer is unusable, the retries spend the same tokens again, and the article stays in
``needs_review`` for good. No DB, no network."""

import pytest

from app.prompts import article_draft
from app.services.quality_review import MODEL_OUTPUT_LIMIT, revision_output_tokens


@pytest.mark.parametrize(("target", "words"), [(1_500, 900), (1_500, 1_500), (1_500, 1_841), (800, 3_000)])  # fmt: skip
def test_a_revision_gets_room_for_the_article_in_hand(target: int, words: int) -> None:
    room = revision_output_tokens(target, words)
    assert room >= article_draft.max_output_tokens(target)  # never less than a fresh draft
    assert room >= words * 5  # the article comes back whole, as JSON, with its changes
    assert room <= MODEL_OUTPUT_LIMIT


def test_a_longer_article_gets_more_room_than_the_target_alone() -> None:
    # The article that stayed stuck: 1,841 words against a 1,500-word target.
    assert revision_output_tokens(1_500, 1_841) > article_draft.max_output_tokens(1_500)
    assert revision_output_tokens(1_500, 900) == revision_output_tokens(1_500, 1_500)


def test_the_request_never_exceeds_what_the_model_accepts() -> None:
    assert revision_output_tokens(1_500, 1_000_000) == MODEL_OUTPUT_LIMIT
