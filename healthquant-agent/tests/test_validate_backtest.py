"""Tests for the reproducible backtest validation CLI helpers."""

from datetime import datetime

import pandas as pd
import pytest

import scripts.validate_backtest as validation


class Cursor(list):
    def sort(self, *_args, **_kwargs):
        return self


class Collection:
    def __init__(self, documents):
        self.documents = documents

    def find(self, *_args, **_kwargs):
        return Cursor(self.documents)


def test_load_predictions_aligns_prices_without_lookahead(monkeypatch) -> None:
    dates = pd.bdate_range("2024-01-02", periods=4)
    db = {
        "regime_states": Collection(
            [
                {"date": day.to_pydatetime(), "predicted_regime": "risk-on"}
                for day in dates
            ]
        )
    }
    price_dates = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame({"Close": [100, 101, 102, 103, 104, 105]}, index=price_dates)
    monkeypatch.setattr(validation.yf, "download", lambda *args, **kwargs: prices)

    frame = validation.load_predictions_from_mongo(db, 2024, 2024)

    assert list(frame.index) == list(dates)
    assert frame["actual_xlv_ret_1d"].iloc[0] == pytest.approx(0.01)
    assert frame["xlv_price"].iloc[-1] == 104


def test_print_metrics_table_includes_disclaimer(capsys) -> None:
    validation.print_metrics_table(
        {
            "total_return_strategy": 0.1,
            "total_return_buyhold": 0.08,
            "max_drawdown_strategy": -0.05,
            "max_drawdown_buyhold": -0.1,
            "sharpe_strategy": 1.0,
            "sharpe_buyhold": 0.7,
            "hit_rate_10d": 0.6,
        }
    )
    output = capsys.readouterr().out
    assert "10-day directional hit" in output
    assert "Past performance does not guarantee future results" in output
