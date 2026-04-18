"""Prompt Lookup Decoding (PLD) draft extraction."""
from __future__ import annotations

import os


def pld_lookup_drafts(
    input_ids,
    generated_ids: list[int],
    *,
    min_match: int = 2,
    max_match: int = 32,
    max_draft: int = 3,
    min_prompt_match_frac: float | None = None,
) -> list[int] | None:
    """
    Find up to `max_draft` draft tokens from the prompt, after a suffix of
    `generated_ids` matches a contiguous span in the prompt.

    Uses the **rightmost** match in the prompt so we prefer continuations from
    the end of the template (e.g. assistant scaffolding), not the first
    occurrence inside the **user question** (which would steal tokens from
    the question text and truncate or derail the answer).

    If `min_prompt_match_frac` is set (default from env PLD_MIN_MATCH_FRAC,
    default 0.55), only matches with start index ``i >= plen * frac`` are
    allowed, so suffixes that only appear in the early user turn are ignored.

    Returns None if no match or not enough continuation tokens in the prompt.
    """
    if len(generated_ids) < min_match:
        return None
    if min_prompt_match_frac is None:
        min_prompt_match_frac = float(os.environ.get("PLD_MIN_MATCH_FRAC", "0.55"))
    min_i = int(len(input_ids) * min_prompt_match_frac)

    p = input_ids
    plen = len(p)
    for k in range(min(len(generated_ids), max_match), min_match - 1, -1):
        suf = generated_ids[-k:]
        # Rightmost i: prefer matches near end of prompt (assistant / scaffold)
        for i in range(plen - k, min_i - 1, -1):
            if i < 0:
                break
            if p[i : i + k] == suf:
                j = i + k - 1
                if j + max_draft <= plen:
                    print(f'-------------------------------------{p[j : j + max_draft]}/{suf}')
                    return p[j : j + max_draft]
    return None


