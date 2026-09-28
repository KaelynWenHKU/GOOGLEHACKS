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


def test_costs_charge_lagged_entry_exit_and_benchmark_entry() -> None:
    predictions = sample_predictions(4)
    predictions["predicted_regime"] = ["risk-on", "neutral", "catalyst-fear", "risk-on"]
    predictions["actual_xlv_ret_1d"] = 0.0
    result = run_backtest(predictions, initial_capital=1000, transaction_cost_bps=100)
    assert result["turnover_strategy"].tolist() == [0, 1, 0.5, 0.5]
    assert result["daily_ret_strategy"].tolist() == pytest.approx([0, -0.01, -0.005, -0.005])
    assert result["strategy_value"].iloc[-1] == pytest.approx(1000 * .99 * .995 ** 2)
    assert result["buyhold_value"].iloc[-1] == pytest.approx(990)
    assert result["daily_ret_buyhold_net"].tolist() == pytest.approx([-.01, 0, 0, 0])
    metrics = compute_metrics(result)
    assert metrics["transaction_cost_bps"] == 100
    assert metrics["total_turnover_strategy"] == 2


def test_turnover_accounts_for_weight_drift_and_cost_precedes_return() -> None:
    predictions = sample_predictions(3)
    predictions["predicted_regime"] = "neutral"
    predictions["actual_xlv_ret_1d"] = [0, .1, 0]
    result = run_backtest(predictions, transaction_cost_bps=100)
    assert result["daily_ret_strategy"].iloc[1] == pytest.approx((1 - .005) * 1.05 - 1)
    assert result["turnover_strategy"].iloc[2] == pytest.approx(.55 / 1.05 - .5)


@pytest.mark.parametrize("bps", [-1, 10000, float("nan"), float("inf")])
def test_invalid_cost_is_rejected(bps) -> None:
    with pytest.raises(ValueError, match="transaction_cost_bps"):
        run_backtest(sample_predictions(), transaction_cost_bps=bps)


def test_costs_do_not_change_raw_market_returns_or_directional_hit_rate() -> None:
    predictions = sample_predictions()
    free = run_backtest(predictions)
    paid = run_backtest(predictions, transaction_cost_bps=20)
    pd.testing.assert_series_equal(free.daily_ret_buyhold, paid.daily_ret_buyhold)
    assert compute_10d_hit_rate(free) == compute_10d_hit_rate(paid)
    assert paid.strategy_value.iloc[-1] < free.strategy_value.iloc[-1]


def test_report_discloses_cost_assumptions(tmp_path) -> None:
    result = run_backtest(sample_predictions(), transaction_cost_bps=12.5)
    path = tmp_path / "costs.html"
    generate_backtest_report(compute_metrics(result), result, str(path))
    report = path.read_text()
    assert "12.5 bps" in report
    assert "No terminal liquidation" in report
