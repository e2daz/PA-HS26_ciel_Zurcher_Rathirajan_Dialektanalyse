"""Regional Distance helpers for Page 2."""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

_VIS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_VIS_DIR))
from _data import (  # noqa: E402
    DAT_COLOR, DIT_COLOR, LAMBDA, filter_preterite, per_sentence_cost,
)

_MODEL_SCALE = alt.Scale(
    domain=["dialect-aware", "dialect-ignorant"],
    range=[DAT_COLOR, DIT_COLOR],
)


def regional_summary(include_praet: bool, dataset: str) -> pd.DataFrame:
    """Per-region mean total cost (DAT, DIT), total cost delta (DIT − DAT), n. Sorted by delta desc."""
    df = filter_preterite(per_sentence_cost(dataset), include_praet)

    summary = (
        df.groupby(["dialect_region", "model"], observed=True)
        .agg(mean_total_cost=("total_cost_per_ref_word", "mean"),
             n_sentences=("path", "size"))
        .reset_index()
    )
    pivoted = summary.pivot(
        index="dialect_region", columns="model",
        values=["mean_total_cost", "n_sentences"],
    )
    out = pd.DataFrame({
        "DAT mean total cost": pivoted[("mean_total_cost", "dialect-aware")],
        "DIT mean total cost": pivoted[("mean_total_cost", "dialect-ignorant")],
        "total cost delta (DIT − DAT)": pivoted[("mean_total_cost", "dialect-ignorant")] - pivoted[
            ("mean_total_cost", "dialect-aware")],
        "n sentences": pivoted[("n_sentences", "dialect-aware")].astype(int),
    })

    # Default ordering: dialect signal strongest first.
    return out.sort_values("total cost delta (DIT − DAT)", ascending=False).reset_index()


def render_intro(uses_balanced: bool) -> None:
    """Page-level explanatory Markdown above the charts."""
    subset_clause = (
        "Computed on the **train_balanced** subset (~25k sentences per region; subset of train_all). "
        if uses_balanced else
        "Computed on the full filtered train_all set; per-region sentence counts vary "
        "(see the *n sentences* column). "
    )
    st.markdown(
        "Dialect distance per region, measured as mean per-sentence **total alignment "
        f"cost** (sum of per-edge costs: substitution = 1 − similarity, ε rows = λ = {LAMBDA}; sum "
        f"divided by reference word count). {subset_clause}Lower total cost = closer to Standard "
        "German; higher **total cost delta (DIT − DAT)** = stronger dialect signal. Regions sorted "
        "by delta (descending)."
    )


def render_headline_plots(summary: pd.DataFrame, regions_sorted: list[str]) -> None:
    """Two charts rendered in sequence: paired DAT/DIT mean total cost bars, then the (DIT − DAT) total cost delta bar."""
    long_summary = summary.melt(
        id_vars=["dialect_region"],
        value_vars=["DAT mean total cost", "DIT mean total cost"],
        var_name="model_label",
        value_name="total_cost",
    )
    long_summary["model"] = long_summary["model_label"].map(
        {"DAT mean total cost": "dialect-aware", "DIT mean total cost": "dialect-ignorant"}
    )

    paired_chart = (
        alt.Chart(long_summary)
        .mark_bar(opacity=0.9)
        .encode(
            x=alt.X("dialect_region:N", sort=regions_sorted, title=None,
                    axis=alt.Axis(labelAngle=0, labelOverlap=False)),
            y=alt.Y("total_cost:Q", title="Mean total cost"),
            color=alt.Color("model:N", scale=_MODEL_SCALE, legend=alt.Legend(title=None)),
            xOffset=alt.XOffset("model:N"),
            tooltip=[
                alt.Tooltip("dialect_region:N", title="Region"),
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("total_cost:Q", format=".4f", title="Mean total cost"),
            ],
        )
        .properties(height=300, title="Mean total cost")
    )

    delta_chart = (
        alt.Chart(summary)
        .mark_bar(opacity=0.9)
        .encode(
            x=alt.X("dialect_region:N", sort=regions_sorted, title=None,
                    axis=alt.Axis(labelAngle=0, labelOverlap=False)),
            y=alt.Y("total cost delta (DIT − DAT):Q", title="Total cost delta (DIT − DAT)"),
            color=alt.condition(
                alt.datum["total cost delta (DIT − DAT)"] >= 0,
                alt.value(DIT_COLOR),
                alt.value(DAT_COLOR),
            ),
            tooltip=[
                alt.Tooltip("dialect_region:N", title="Region"),
                alt.Tooltip("total cost delta (DIT − DAT):Q", format=".4f", title="Total cost delta (DIT − DAT)"),
                alt.Tooltip("n sentences:Q", title="N sentences"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeWidth=0.6, color="black").encode(y="y:Q")

    # Render as separate charts (not vconcat) so each picks up the full container width.
    st.altair_chart(paired_chart, width="stretch")
    st.markdown("**Total cost delta**")
    st.altair_chart((delta_chart + zero).properties(height=220), width="stretch")


def render_summary_table(summary: pd.DataFrame, uses_balanced: bool) -> None:
    """Per-region summary table with rounded total cost columns."""
    st.markdown("### Per-region summary")
    subset_clause = "all train_balanced sentences" if uses_balanced else "all filtered sentences"
    st.caption(f"Dialect distance per region: per-sentence total alignment cost averaged across {subset_clause} "
               "in the region, plus the DIT-vs-DAT delta.")
    display_table = summary.copy()

    for col in ("DAT mean total cost", "DIT mean total cost", "total cost delta (DIT − DAT)"):
        display_table[col] = display_table[col].round(4)

    n_help = (
        "Number of sentences from the train_balanced subset contributing to this region's means."
        if uses_balanced else
        "Number of sentences contributing to this region's means (varies; no balanced subset)."
    )
    column_config = {
        "DAT mean total cost": st.column_config.NumberColumn(
            help="Mean DAT total cost: average per-sentence total alignment cost across all sentences in this region.",
        ),
        "DIT mean total cost": st.column_config.NumberColumn(
            help="Mean DIT total cost: average per-sentence total alignment cost across all sentences in this region.",
        ),
        "total cost delta (DIT − DAT)": st.column_config.NumberColumn(
            help="Total cost delta: DIT mean total cost − DAT mean total cost.",
        ),
        "n sentences": st.column_config.NumberColumn(help=n_help),
    }

    st.dataframe(display_table, hide_index=True, width="stretch", column_config=column_config)
    st.caption("Default ordering: descending by **total cost delta (DIT − DAT)**.")


def render_cost_distribution(per_sentence: pd.DataFrame, regions_sorted: list[str]) -> None:
    """Per-region/model box plot of per-sentence total alignment cost (5/25/50/75/95 percentiles)."""
    st.markdown("### Per-sentence total cost distribution")
    st.caption(
        "Each box: within-region distribution of per-sentence total alignment cost (per ref word). "
        "Whiskers span 5th–95th percentile; box is q1–q3; tick is the median. "
        "Wide IQR = mixed dialect compliance across speakers."
    )

    box_stats = _box_stats(per_sentence)
    base = alt.Chart(box_stats).encode(
        x=alt.X("dialect_region:N", sort=regions_sorted, title=None,
                axis=alt.Axis(labelAngle=0, labelOverlap=False)),
        color=alt.Color("model:N", scale=_MODEL_SCALE, legend=alt.Legend(title=None)),
        xOffset=alt.XOffset("model:N"),
    )
    whisker = base.mark_rule(strokeWidth=1.5).encode(
        y=alt.Y("min:Q", title="Total cost per ref word"),
        y2="max:Q",
    )
    box = base.mark_bar(size=22, opacity=0.9).encode(y="q1:Q", y2="q3:Q")
    median_tick = base.mark_tick(thickness=2.5, color="black", size=22).encode(y="median:Q")

    st.altair_chart((whisker + box + median_tick).properties(height=380), width="stretch")


def _box_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-sentence total cost into box-plot quantiles per (region, model)."""
    out = []
    for (region, model), grp in df.groupby(["dialect_region", "model"], observed=True):
        v = grp["total_cost_per_ref_word"]
        out.append({
            "dialect_region": region, "model": model,
            "min": v.quantile(0.05), "q1": v.quantile(0.25),
            "median": v.median(),
            "q3": v.quantile(0.75), "max": v.quantile(0.95),
        })
    return pd.DataFrame(out)
