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


def price_validation_db():
    """A single verified prediction for testing market-provider integrity."""
    return {"regime_states": Collection([
        {"date": datetime(2024, 1, 2), "predicted_regime": "risk-on",
         "train_end_date": "2023-12-29"}
    ])}


@pytest.mark.parametrize("bad_price", [0, -100, float("inf"), float("-inf"), float("nan"), "invalid"])
def test_rejects_invalid_prices_before_return_calculation(monkeypatch, bad_price):
    prices = pd.DataFrame({"Close": [bad_price, 101]},
                          index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    monkeypatch.setattr(validation.yf, "download", lambda *a, **k: prices)
    with pytest.raises(ValueError, match="prices"):
        validation.load_predictions_from_mongo(price_validation_db(), 2024, 2024)


@pytest.mark.parametrize("dates", [
    ["2024-01-01", "2024-01-01", "2024-01-02"],
    [None, "2024-01-01", "2024-01-02"],
])
def test_rejects_ambiguous_price_dates(monkeypatch, dates):
    prices = pd.DataFrame({"Close": [99, 100, 101]}, index=pd.to_datetime(dates))
    monkeypatch.setattr(validation.yf, "download", lambda *a, **k: prices)
    with pytest.raises(ValueError, match="price dates"):
        validation.load_predictions_from_mongo(price_validation_db(), 2024, 2024)


def test_unsorted_prices_are_sorted_before_return_calculation(monkeypatch):
    prices = pd.DataFrame({"Close": [101, 100]},
                          index=pd.to_datetime(["2024-01-02", "2024-01-01"]))
    monkeypatch.setattr(validation.yf, "download", lambda *a, **k: prices)
    result = validation.load_predictions_from_mongo(price_validation_db(), 2024, 2024)
    assert result.actual_xlv_ret_1d.iloc[0] == pytest.approx(.01)


def test_rejects_duplicate_price_dates_after_timezone_removal(monkeypatch):
    # Two UTC instants collapse to the same wall-clock timestamp at DST fallback.
    dates = pd.to_datetime(["2024-11-03T05:30:00Z", "2024-11-03T06:30:00Z"]).tz_convert("America/New_York")
    prices = pd.DataFrame({"Close": [100, 101]}, index=dates)
    monkeypatch.setattr(validation.yf, "download", lambda *a, **k: prices)
    with pytest.raises(ValueError, match="price dates"):
        validation.load_predictions_from_mongo(price_validation_db(), 2024, 2024)


def test_load_predictions_aligns_prices_without_lookahead(monkeypatch) -> None:
    dates = pd.bdate_range("2024-01-02", periods=4)
    db = {
        "regime_states": Collection(
            [
                {"date": day.to_pydatetime(), "predicted_regime": "risk-on", "train_end_date": "2023-12-29"}
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


@pytest.mark.parametrize("extra", [
    {},
    {"train_end_date": "2024-01-02"},
    {"train_end_date": "2024-01-03"},
])
def test_rejects_unverified_or_in_sample_predictions(extra, monkeypatch):
    """Reject invalid provenance before accessing any market provider."""
    document = {"date": datetime(2024, 1, 2), "predicted_regime": "risk-on", **extra}
    def unexpected_download(*args, **kwargs):
        pytest.fail("Invalid predictions must fail before downloading prices")
    monkeypatch.setattr(validation.yf, "download", unexpected_download)
    with pytest.raises(ValueError, match="train_end_date"):
        validation.load_predictions_from_mongo({"regime_states": Collection([document])}, 2024, 2024)


def test_rejects_missing_trading_session_predictions(monkeypatch):
    documents = [
        {"date": datetime(2024, 1, day), "predicted_regime": "risk-on",
         "train_end_date": "2023-12-29"} for day in (2, 4)
    ]
    prices = pd.DataFrame({"Close": [100, 101, 102, 103]},
                          index=pd.bdate_range("2024-01-01", periods=4))
    monkeypatch.setattr(validation.yf, "download", lambda *args, **kwargs: prices)
    with pytest.raises(ValueError, match="Missing predictions"):
        validation.load_predictions_from_mongo({"regime_states": Collection(documents)}, 2024, 2024)


def test_main_forwards_cost_and_persists_assumptions(monkeypatch, tmp_path, capsys):
    """Exercise the real simulator/report writer without any external services."""
    import json
    import database.mongo_client as mongo

    predictions = pd.DataFrame({
        "predicted_regime": ["risk-on"] * 12,
        "actual_xlv_ret_1d": [0.0] * 12,
    }, index=pd.bdate_range("2024-01-02", periods=12))
    monkeypatch.setattr(mongo, "get_db", lambda: {})
    monkeypatch.setattr(validation, "load_predictions_from_mongo", lambda *_: predictions)
    monkeypatch.setattr(validation, "OUTPUT_DIR", tmp_path)
    validation.main(2024, 2024, transaction_cost_bps=25)
    metrics = json.loads((tmp_path / "backtest_metrics.json").read_text())
    assert metrics["transaction_cost_bps"] == 25
    assert metrics["total_return_strategy"] == pytest.approx(-.0025)
    assert metrics["total_return_buyhold"] == pytest.approx(-.0025)
    assert "25 bps" in capsys.readouterr().out
    assert "25 bps" in (tmp_path / "backtest_report.html").read_text()


def test_main_rejects_invalid_cost_before_connecting(monkeypatch):
    import database.mongo_client as mongo

    def unexpected_connection():
        pytest.fail("Invalid configuration must not access MongoDB")
    monkeypatch.setattr(mongo, "get_db", unexpected_connection)
    with pytest.raises(ValueError, match="transaction_cost_bps"):
        validation.main(transaction_cost_bps=-1)
