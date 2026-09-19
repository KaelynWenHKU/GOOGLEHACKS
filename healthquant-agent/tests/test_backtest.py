"""Tests for the no-lookahead HealthQuant portfolio backtest."""

import numpy as np
import pandas as pd
import pytest

from hmm.backtest import (
    compute_10d_hit_rate,
    compute_metrics,
    generate_backtest_report,
    plot_regime_timeline,
    run_backtest,
)


def sample_predictions(rows: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=rows)
    regimes = (["risk-on"] * 12 + ["catalyst-fear"] * 12 + ["neutral"] * 12)[:rows]
    returns = np.array([0.01] * 13 + [-0.01] * 12 + [0.002] * 12)[:rows]
    return pd.DataFrame(
        {
            "predicted_regime": regimes,
            "actual_xlv_ret_1d": returns,
            "xlv_price": 100 * np.cumprod(1 + returns),
        },
        index=dates,
    )


def test_backtest_lags_signal_one_session_and_compounds() -> None:
    result = run_backtest(sample_predictions(), initial_capital=1_000)
    assert result["executed_weight"].iloc[0] == 0.0
    assert result["executed_weight"].iloc[1] == 1.0
    assert result["daily_ret_strategy"].iloc[0] == 0.0
    assert result["strategy_value"].iloc[1] == 1_010.0
    assert result["drawdown_strategy"].le(0).all()


def test_metrics_are_finite_and_hit_rate_uses_future_ten_days() -> None:
    result = run_backtest(sample_predictions())
    metrics = compute_metrics(result)
    assert metrics["trading_days"] == 30
    assert np.isfinite(metrics["sharpe_strategy"])
    assert 0.0 <= metrics["hit_rate_10d"] <= 1.0
    assert compute_10d_hit_rate(result) == metrics["hit_rate_10d"]


def test_timeline_and_report_are_written_without_network(tmp_path) -> None:
    predictions = sample_predictions()
    result = run_backtest(predictions)
    metrics = compute_metrics(result)
    timeline_path = tmp_path / "timeline.html"
    report_path = tmp_path / "report.html"

    figure = plot_regime_timeline(predictions, str(timeline_path))
    generate_backtest_report(metrics, result, str(report_path))

    assert figure.data
    assert timeline_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Past performance does not guarantee future results" in report
    assert "Out-of-Sample" in report


def test_drawdown_includes_initial_capital_on_first_loss() -> None:
    """A loss before the first observed equity high must still count."""
    predictions = sample_predictions(3)
    predictions["actual_xlv_ret_1d"] = [-0.1, -0.1, 0.05]
    result = run_backtest(predictions, initial_capital=500)
    assert result["drawdown_buyhold"].iloc[0] == pytest.approx(-0.1)
    assert compute_metrics(result)["max_drawdown_buyhold"] == pytest.approx(-0.19)


@pytest.mark.parametrize("capital", [float("nan"), float("inf"), 0, -1])
def test_invalid_capital_is_rejected(capital) -> None:
    with pytest.raises(ValueError, match="capital"):
        run_backtest(sample_predictions(), initial_capital=capital)


def test_missing_regime_is_rejected_instead_of_becoming_cash() -> None:
    predictions = sample_predictions()
    predictions.iloc[2, predictions.columns.get_loc("predicted_regime")] = None
    with pytest.raises(ValueError, match="regime"):
        run_backtest(predictions)


def test_flat_forward_return_is_not_a_correct_fear_prediction() -> None:
    predictions = sample_predictions(12)
    predictions["predicted_regime"] = "catalyst-fear"
    predictions["actual_xlv_ret_1d"] = 0.0
    assert compute_10d_hit_rate(run_backtest(predictions)) == 0.0
