"""Quick EDA on the weekly demand table. Answers four specific questions."""

from pathlib import Path

import pandas as pd



DATA = Path(__file__).resolve().parents[2] / "data" / "processed" / "weekly_demand.csv"


def main() -> None:
    df = pd.read_csv(DATA, parse_dates=["week_start"])

    print("=" * 60)
    print("Q1: MONTHLY SEASONALITY")
    print("=" * 60)
    monthly = df.groupby(df["week_start"].dt.month)["units_sold"].sum()
    avg = monthly.mean()
    for month, units in monthly.items():
        bar = "#" * int(40 * units / monthly.max())
        flag = "  <-- PEAK" if units > avg * 1.3 else ""
        print(f"{month:2d} | {units:>8,} {bar}{flag}")

    print()
    print("=" * 60)
    print("Q2: TOTAL WEEKLY DEMAND")
    print("=" * 60)
    wt = df.groupby("week_start")["units_sold"].sum()
    print(f"Mean : {wt.mean():,.0f}")
    print(f"Min  : {wt.min():,.0f} ({wt.idxmin().date()})")
    print(f"Max  : {wt.max():,.0f} ({wt.idxmax().date()})")
    print(f"CV   : {wt.std() / wt.mean():.2f}")

    print()
    print("=" * 60)
    print("Q3: PER-PRODUCT VOLATILITY")
    print("=" * 60)
    stats = df.groupby("product_id")["units_sold"].agg(["mean", "std", "sum"])
    stats["cv"] = stats["std"] / stats["mean"]
    print(f"Median CV: {stats['cv'].median():.2f}")
    print()
    print("5 most stable:")
    print(stats.nsmallest(5, "cv")[["mean", "cv"]].round(2).to_string())
    print()
    print("5 most volatile:")
    print(stats.nlargest(5, "cv")[["mean", "cv"]].round(2).to_string())

    print()
    print("=" * 60)
    print("Q4: TREND (year 1 vs year 2)")
    print("=" * 60)
    split = pd.Timestamp("2010-12-06")
    y1 = df[df["week_start"] < split]["units_sold"].sum()
    y2 = df[df["week_start"] >= split]["units_sold"].sum()
    print(f"Year 1: {y1:,}")
    print(f"Year 2: {y2:,}")
    print(f"Change: {100 * (y2 - y1) / y1:+.1f}%")


if __name__ == "__main__":
    main()


