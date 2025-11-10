"""Scoring functions for probabilistic forecasting."""


def quantile_loss(
    y: float,
    q_value: float,
    q_level: float,
) -> float:
    r"""
    Calculate the quantile loss (pinball loss).

    The quantile loss is defined as:
    .. math::
        L_\alpha(y, q) = (q - y) \cdot (\mathbb{1}(y \leq q) - \alpha)

    where:
        - y is the observed value (`y`)
        - q is the predicted quantile value (`q_value`)
        - α is the quantile level (`q_level`)

    Parameters
    ----------
    y : float
        True value observed.
    q_value : float
        Predicted quantile values.
    q_level : float
        Quantile level, must be between 0 and 1.

    Returns
    -------
    float
        Quantile loss value.
    """
    indicator = 1 if y <= q_value else 0
    return (q_value - y) * (indicator - q_level)
