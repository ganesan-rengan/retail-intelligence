"""Pydantic request/response models for the forecasting API."""

from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator

SUPPORTED_HORIZON_DAYS = 28


class ForecastRequest(BaseModel):
    """Input for a single-product forecast request."""

    product_id: str = Field(
        ...,
        description="Identifier of the product to forecast.",
        examples=["49"],
    )
    horizon_days: int = Field(
        default=SUPPORTED_HORIZON_DAYS,
        description="Forecast horizon in days. Only 28 (4 weeks) is supported.",
        examples=[28],
    )

    @field_validator("horizon_days")
    @classmethod
    def reject_unsupported_horizon(cls, value: int) -> int:
        if value != SUPPORTED_HORIZON_DAYS:
            raise ValueError(
                f"horizon_days={value} is not supported; the model only "
                f"produces {SUPPORTED_HORIZON_DAYS}-day (4-week) forecasts. "
                "Multi-horizon support is future scope."
            )
        return value


class ForecastResponse(BaseModel):
    """A single-product forecast for the fixed 4-week horizon."""

    product_id: str = Field(
        ...,
        description="Identifier of the forecasted product.",
        examples=["49"],
    )
    predicted_units: float = Field(
        ...,
        ge=0,
        description="Predicted total units for the horizon window. Cannot be negative.",
        examples=[123.45],
    )
    week_start: date = Field(
        ...,
        description="First day of the forecast window.",
        examples=["2026-09-14"],
    )
    week_end: date = Field(
        ...,
        description="Last day of the forecast window.",
        examples=["2026-10-11"],
    )
    model_version: str = Field(
        ...,
        description="Version of the model that produced this forecast.",
        examples=["v3"],
    )
    horizon_days: int = Field(
        ...,
        description="Forecast horizon in days.",
        examples=[28],
    )
    generated_at: datetime = Field(
        ...,
        description="UTC timestamp when this forecast was generated.",
        examples=["2026-09-14T12:00:00Z"],
    )


class HealthResponse(BaseModel):
    """Readiness state of the service. Used for both the 200 and 503 outcomes."""

    status: str = Field(
        ...,
        description="Overall readiness: 'healthy' or 'unhealthy'.",
        examples=["healthy"],
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the forecasting model was loaded at startup.",
        examples=[True],
    )
    database: str = Field(
        ...,
        description="Database connectivity: 'up' or 'down'.",
        examples=["up"],
    )
    model_version: str | None = Field(
        default=None,
        description="Version of the loaded model, or null if none is loaded.",
        examples=["v3"],
    )


class ModelInfoResponse(BaseModel):
    """Metadata about the currently loaded model artifact."""

    model_version: str = Field(
        ...,
        description="Version identifier of the loaded model.",
        examples=["v3"],
    )
    trained_at: datetime = Field(
        ...,
        description="UTC timestamp when the model was trained.",
        examples=["2026-09-12T03:19:26.596680+00:00"],
    )
    algorithm: str = Field(
        ...,
        description="Learning algorithm used to train the model.",
        examples=["lightgbm"],
    )
    horizon_weeks: int = Field(
        ...,
        description="Forecast horizon in weeks.",
        examples=[4],
    )
    n_products: int = Field(
        ...,
        description="Number of products the model was trained to forecast.",
        examples=[50],
    )
    n_features: int = Field(
        ...,
        description="Number of input features used by the model.",
        examples=[21],
    )
    best_baseline: str = Field(
        ...,
        description="Name of the strongest non-model baseline compared against.",
        examples=["4-week moving average"],
    )
    best_baseline_wape: float = Field(
        ...,
        description="WAPE of the best baseline on the test period.",
        examples=[0.766],
    )
    test_wape: float = Field(
        ...,
        description="WAPE of the model on the test period. Headline accuracy metric.",
        examples=[0.703],
    )
    improvement_points: float = Field(
        ...,
        description="Absolute WAPE improvement over the best baseline, in points.",
        examples=[0.063],
    )
    improvement_relative: float = Field(
        ...,
        description="Relative WAPE improvement over the best baseline.",
        examples=[0.082],
    )


class ProductsResponse(BaseModel):
    """The set of products the model can forecast."""

    products: list[str] = Field(
        ...,
        description="Identifiers of all products the model can forecast.",
        examples=[["1", "2", "3"]],
    )
    count: int = Field(
        ...,
        description="Number of products in the list.",
        examples=[50],
    )


class ErrorResponse(BaseModel):
    """A caller-facing error message, e.g. for a bad or unknown product_id."""

    detail: str = Field(
        ...,
        description="Human-readable explanation of what went wrong.",
        examples=["Product '999' not found."],
    )
