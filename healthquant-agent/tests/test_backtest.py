"""Tests for the no-lookahead HealthQuant portfolio backtest."""

import numpy as np
import pandas as pd

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

