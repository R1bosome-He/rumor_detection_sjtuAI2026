"""LLM explanation module — generates natural-language explanations for predictions."""


def explain(text: str, label: int, evidence: list) -> str | None:
    """Generate an explanation for a rumor detection result.

    Parameters
    ----------
    text : str
        The original input text.
    label : int
        Predicted label (0 = non-rumor, 1 = rumor).
    evidence : list of str
        Key evidence words from the model's attention.

    Returns
    -------
    str or None
        Natural-language explanation, or None if not yet implemented.
    """
    # TODO: integrate LLM for explanation generation
    return None
