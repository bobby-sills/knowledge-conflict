"""Document construction and prompt assembly.

Two document variants per fact, from one fixed template:

  faithful  — states the gold object
  corrupted — states a distractor in the gold's place

**Limitation, stated out loud as the spec requires:** these are synthetic
template passages, not retrieved documents. They are cleaner, shorter and more
assertive than anything a retriever returns, which inflates how much authority
the context appears to carry and removes the partial/hedged/multi-claim cases
that make real conflict messy. The pilot accepts that; the paper cannot. The
replacement candidates are ClashEval, ConflictQA and ConFiQA — flagged in
DECISIONS.md rather than silently chosen.

One template, filled per relation, keeps document length and assertiveness
constant between faithful and corrupted variants. If the corrupted passage were
even slightly less fluent, every "resistance" result would be confounded by
fluency rather than by conflict.
"""

from __future__ import annotations

from . import config

# PopQA's 16 relations. `{s}` = subject, `{o}` = object.
RELATION_TEMPLATES: dict[str, str] = {
    "occupation":        "{s} worked as a {o}. Contemporary records and biographical directories list {o} as {s}'s profession.",
    "place of birth":    "{s} was born in {o}. Civil registers for {o} record the birth, and later biographies repeat the {o} birthplace.",
    "genre":             "{s} belongs to the {o} genre. Critics and catalogues file {s} under {o}.",
    "father":            "{s}'s father was {o}. Family records name {o} as the father of {s}.",
    "country":           "{s} is located in {o}. Administrative records place {s} within the borders of {o}.",
    "producer":           "{s} was produced by {o}. Production credits list {o} as the producer of {s}.",
    "director":          "{s} was directed by {o}. The directing credit for {s} belongs to {o}.",
    "capital of":        "{s} is the capital of {o}. Government functions of {o} are seated in {s}.",
    "screenwriter":      "{s} was written by {o}. The screenplay for {s} is credited to {o}.",
    "composer":          "The music for {s} was composed by {o}. The score of {s} is credited to {o}.",
    "color":             "{s} is {o} in colour. Descriptions of {s} consistently give its colour as {o}.",
    "religion":          "{s} followed {o}. Contemporary accounts describe {s} as an adherent of {o}.",
    "sport":             "{s} competed in {o}. Career summaries of {s} list {o} as the sport.",
    "author":            "{s} was written by {o}. The authorship of {s} is attributed to {o}.",
    "mother":            "{s}'s mother was {o}. Family records name {o} as the mother of {s}.",
    "capital":           "The capital of {s} is {o}. {o} houses the seat of government of {s}.",
}

FALLBACK_TEMPLATE = (
    "{s} is associated with {o}. Reference works give {o} as the {r} of {s}.")

PREAMBLE = "{title} is the subject of the following encyclopedia extract.\n\n"


def make_document(fact: dict, variant: str, distractor_index: int = 0) -> tuple[str, str]:
    """Return (document_text, stated_object).

    `variant` is "faithful" or "corrupted". The corrupted variant substitutes
    `fact["distractors"][distractor_index]` — index 0 by default so the document
    is a deterministic function of the fact and the variant, and a rerun after a
    dead session produces byte-identical text.
    """
    if variant == "faithful":
        stated = fact["object"]
    elif variant == "corrupted":
        if not fact.get("distractors"):
            raise ValueError(f"{fact['fact_id']} has no distractor to corrupt with")
        stated = fact["distractors"][distractor_index % len(fact["distractors"])]
    else:
        raise ValueError(f"unknown document variant {variant!r}")

    template = RELATION_TEMPLATES.get(fact["relation"], FALLBACK_TEMPLATE)
    body = template.format(s=fact["subject"], o=stated, r=fact["relation"])
    title = fact.get("s_wiki_title") or fact["subject"]
    return PREAMBLE.format(title=title) + body, stated


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

def _chat(tokenizer, user_text: str) -> str:
    """Render a single user turn with the model's own chat template.

    `add_generation_prompt=True` leaves the string ending at the assistant
    header, so the next token the model produces is the first token of the
    answer. Every score in this study is read at that position, in both passes,
    which is what makes the prior and context distributions comparable.
    """
    msgs = []
    if config.SYSTEM_PROMPT:
        msgs.append({"role": "system", "content": config.SYSTEM_PROMPT})
    msgs.append({"role": "user", "content": user_text})
    return tokenizer.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)


def prior_prompt(fact: dict, tokenizer=None, style: str | None = None) -> str:
    """Question only — no document. Gives p_pri."""
    style = style or config.PROMPT_STYLE
    if fact.get("source") == "tristate" and fact.get("prompt_pri_raw"):
        return fact["prompt_pri_raw"]
    q = fact["question"]
    if style == "raw":
        return f"Answer the following question: \nQuestion: {q}\nAnswer:"
    if tokenizer is None:
        raise ValueError("chat prompt style needs a tokenizer")
    return _chat(tokenizer, f"Question: {q}")


def context_prompt(fact: dict, document: str, tokenizer=None,
                   style: str | None = None) -> str:
    """Document + question. Gives p_ctx."""
    style = style or config.PROMPT_STYLE
    q = fact["question"]
    if style == "raw":
        return (f"{document} \n{config.CTX_INSTRUCTION} \n"
                f"Question: {q}\nAnswer:")
    if tokenizer is None:
        raise ValueError("chat prompt style needs a tokenizer")
    return _chat(tokenizer, f"{document}\n\n{config.CTX_INSTRUCTION}\nQuestion: {q}")


def build_conflict_cases(facts: list[dict], states: dict[str, str]) -> list[dict]:
    """Expand labelled facts into the (fact, document) cases each test needs.

    From Test 0's state table:
        prior wrong   + faithful  -> correction
        prior correct + corrupted -> resistance
        prior correct + faithful  -> agreement
    Ambiguous priors produce no case (kept in the counts, dropped from analysis).
    """
    cases = []
    for fact in facts:
        prior = states.get(fact["fact_id"])
        if prior == "wrong":
            variants = [("faithful", "correction")]
        elif prior == "correct":
            variants = [("corrupted", "resistance"), ("faithful", "agreement")]
        else:
            continue
        for variant, state in variants:
            if variant == "corrupted" and not fact.get("distractors"):
                continue
            doc, stated = make_document(fact, variant)
            cases.append({
                "case_id": f"{fact['fact_id']}::{variant}",
                "fact_id": fact["fact_id"],
                "split": fact["split"],
                "state": state,
                "doc_variant": variant,
                "document": doc,
                "stated_object": stated,
                "prior_label": prior,
            })
    return cases
