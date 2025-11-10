"""Tests for scoring functions."""

from prob4cast import quantile_loss


def test_quantile_loss():
    """Test quantile loss for median (q_level=0.5)."""
    # When y > q_value, loss should be q_level * (y - q_value)
    assert quantile_loss(y=10.0, q_value=10.0, q_level=0.5) == 0.0