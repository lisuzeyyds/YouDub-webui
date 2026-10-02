from __future__ import annotations


class DeterministicStageError(RuntimeError):
    """A stage failure that will not disappear by retrying it.

    Examples: the translation endpoint reports an exhausted balance, the API key
    is rejected, the configured model does not exist. The pipeline must surface
    the message instead of burning attempts, because each attempt costs an API
    call and the outcome is already known.
    """
