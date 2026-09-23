"""FastAPI endpoint tests. Every DB touch goes through dependency_overrides
on get_db (now shared by /health and /forecast); every model touch goes
through a fake model object placed in main.state. No real database or model
artifact is used -- see conftest.py.
"""
from datetime import date
from unittest.mock import MagicMock

from conftest import TRAINED_PRODUCTS, Row

from forecasting_service.app import main


class TestHealth:
    def test_returns_200_when_model_loaded_and_db_ok(self, api_client, fake_model):
        main.state["model"] = fake_model
        main.state["metrics"] = {"model_version": "vTEST"}
        main.app.dependency_overrides[main.get_db] = lambda: MagicMock()

        resp = api_client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True
        assert body["database"] == "up"

    def test_returns_503_when_model_not_loaded(self, api_client):
        main.app.dependency_overrides[main.get_db] = lambda: MagicMock()

        resp = api_client.get("/health")

        assert resp.status_code == 503
        assert resp.json()["model_loaded"] is False

    def test_returns_503_when_database_unavailable(self, api_client, fake_model):
        main.state["model"] = fake_model
        main.state["metrics"] = {"model_version": "vTEST"}
        failing_db = MagicMock()
        failing_db.execute.side_effect = RuntimeError("connection refused")
        main.app.dependency_overrides[main.get_db] = lambda: failing_db

        resp = api_client.get("/health")

        assert resp.status_code == 503
        assert resp.json()["database"] == "down"


class TestForecast:
    def _override_db(self, latest_week_rows, sales_rows):
        db = MagicMock()
        db.execute.side_effect = [
            MagicMock(all=MagicMock(return_value=latest_week_rows)),
            MagicMock(all=MagicMock(return_value=sales_rows)),
        ]
        main.app.dependency_overrides[main.get_db] = lambda: db

    def test_returns_200_for_a_trained_product(self, api_client, fake_model, sample_metrics):
        main.state["model"] = fake_model
        main.state["metrics"] = sample_metrics
        self._override_db(
            latest_week_rows=[Row(date(2011, 11, 21)), Row(date(2011, 11, 28))],
            sales_rows=[Row(date(2011, 11, 28), 100)],
        )

        resp = api_client.post("/forecast", json={"product_id": TRAINED_PRODUCTS[0]})

        assert resp.status_code == 200
        body = resp.json()
        assert body["product_id"] == TRAINED_PRODUCTS[0]
        assert body["predicted_units"] == fake_model._prediction
        assert body["week_start"] == "2011-12-26"
        assert body["week_end"] == "2012-01-01"
        assert body["horizon_days"] == 28

    def test_returns_404_for_an_untrained_product(self, api_client, fake_model, sample_metrics):
        main.state["model"] = fake_model
        main.state["metrics"] = sample_metrics
        # build_forecast checks the trained-product list before ever touching
        # the database -- but get_db is still overridden below, because
        # FastAPI resolves every route dependency on every request whether
        # or not the handler body ends up using it.
        main.app.dependency_overrides[main.get_db] = lambda: MagicMock()

        resp = api_client.post("/forecast", json={"product_id": "NOT-A-REAL-PRODUCT"})

        assert resp.status_code == 404
        assert "NOT-A-REAL-PRODUCT" in resp.json()["detail"]

    def test_returns_422_for_an_unsupported_horizon(self, api_client, fake_model, sample_metrics):
        main.state["model"] = fake_model
        main.state["metrics"] = sample_metrics
        main.app.dependency_overrides[main.get_db] = lambda: MagicMock()

        resp = api_client.post(
            "/forecast", json={"product_id": TRAINED_PRODUCTS[0], "horizon_days": 7}
        )

        assert resp.status_code == 422

    def test_returns_422_when_product_id_is_missing(self, api_client, fake_model, sample_metrics):
        main.state["model"] = fake_model
        main.state["metrics"] = sample_metrics
        main.app.dependency_overrides[main.get_db] = lambda: MagicMock()

        resp = api_client.post("/forecast", json={})

        assert resp.status_code == 422


class TestModelInfo:
    def test_returns_all_expected_fields(self, api_client, fake_model, sample_metrics):
        main.state["model"] = fake_model
        main.state["metrics"] = sample_metrics

        resp = api_client.get("/model-info")

        assert resp.status_code == 200
        body = resp.json()
        expected_fields = {
            "model_version", "trained_at", "algorithm", "horizon_weeks",
            "n_products", "n_features", "best_baseline", "best_baseline_wape",
            "test_wape", "improvement_points", "improvement_relative",
        }
        assert expected_fields <= body.keys()
        assert body["model_version"] == sample_metrics["model_version"]
        assert body["n_features"] == len(sample_metrics["features"])
        assert body["test_wape"] == sample_metrics["metrics"]["test_wape"]


class TestProducts:
    def test_count_matches_product_list_length(self, api_client, fake_model):
        main.state["model"] = fake_model

        resp = api_client.get("/products")

        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == len(body["products"])
        assert body["count"] == len(TRAINED_PRODUCTS)
        assert set(body["products"]) == set(TRAINED_PRODUCTS)


class TestRoot:
    def test_redirects_to_docs(self, api_client):
        resp = api_client.get("/", follow_redirects=False)
        assert resp.status_code == 307
        assert resp.headers["location"] == "/docs"
