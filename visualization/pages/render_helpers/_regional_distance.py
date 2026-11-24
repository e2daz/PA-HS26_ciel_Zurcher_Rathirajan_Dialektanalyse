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
from _help import chart_help  # noqa: E402
from i18n import t  # noqa: E402

_MODEL_KEYS = {"dialect-aware": "regional.model_dat", "dialect-ignorant": "regional.model_dit"}


def _model_labels(models: pd.Series) -> pd.Series:
    """The model names as shown in the charts' legend and tooltips, in the current language."""
    return models.astype(str).map(lambda model: t(_MODEL_KEYS[model]))


def _model_scale() -> alt.Scale:
    """DAT and DIT in their fixed colours, keyed by the translated model names."""
    return alt.Scale(domain=[t("regional.model_dat"), t("regional.model_dit")], range=[DAT_COLOR, DIT_COLOR])


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
    subset = t("regional.intro_balanced") if uses_balanced else t("regional.intro_unbalanced")
    st.markdown(t("regional.intro", lambda_=LAMBDA, subset=subset))


def render_headline_plots(summary: pd.DataFrame, regions_sorted: list[str]) -> None:
    """Two charts rendered in sequence: paired DAT/DIT mean total cost bars, then the (DIT − DAT) total cost delta bar."""
    long_summary = summary.melt(
        id_vars=["dialect_region"],
        value_vars=["DAT mean total cost", "DIT mean total cost"],
        var_name="model_label",
        value_name="total_cost",
    )
    long_summary["model"] = _model_labels(long_summary["model_label"].map(
        {"DAT mean total cost": "dialect-aware", "DIT mean total cost": "dialect-ignorant"}
    ))

    paired_chart = (
        alt.Chart(long_summary)
        .mark_bar(opacity=0.9)
        .encode(
            x=alt.X("dialect_region:N", sort=regions_sorted, title=None,
                    axis=alt.Axis(labelAngle=0, labelOverlap=False)),
            y=alt.Y("total_cost:Q", title=t("regional.mean_total_cost")),
            color=alt.Color("model:N", scale=_model_scale(), legend=alt.Legend(title=None)),
            xOffset=alt.XOffset("model:N"),
            tooltip=[
                alt.Tooltip("dialect_region:N", title=t("regional.region")),
                alt.Tooltip("model:N", title=t("regional.model")),
                alt.Tooltip("total_cost:Q", format=".4f", title=t("regional.mean_total_cost")),
            ],
        )
        .properties(height=300, title=t("regional.mean_total_cost"))
    )

    delta_chart = (
        alt.Chart(summary)
        .mark_bar(opacity=0.9)
        .encode(
            x=alt.X("dialect_region:N", sort=regions_sorted, title=None,
                    axis=alt.Axis(labelAngle=0, labelOverlap=False)),
            y=alt.Y("total cost delta (DIT − DAT):Q", title=t("regional.delta_axis")),
            color=alt.condition(
                alt.datum["total cost delta (DIT − DAT)"] >= 0,
                alt.value(DIT_COLOR),
                alt.value(DAT_COLOR),
            ),
            tooltip=[
                alt.Tooltip("dialect_region:N", title=t("regional.region")),
                alt.Tooltip("total cost delta (DIT − DAT):Q", format=".4f", title=t("regional.delta_axis")),
                alt.Tooltip("n sentences:Q", title=t("regional.n_sentences")),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeWidth=0.6, color="black").encode(y="y:Q")

    # Render as separate charts (not vconcat) so each picks up the full container width.
    st.altair_chart(paired_chart, width="stretch")
    chart_help(t("regional.paired_shows"), t("regional.paired_use"))
    st.markdown(f"**{t('regional.delta_title')}**")
    st.altair_chart((delta_chart + zero).properties(height=220), width="stretch")
    chart_help(t("regional.delta_shows"), t("regional.delta_use"))


def render_summary_table(summary: pd.DataFrame, uses_balanced: bool) -> None:
    """Per-region summary table with rounded total cost columns."""
    st.markdown(f"### {t('regional.summary_title')}")
    subset = t("regional.summary_balanced") if uses_balanced else t("regional.summary_unbalanced")
    display_table = summary.copy()

    for col in ("DAT mean total cost", "DIT mean total cost", "total cost delta (DIT − DAT)"):
        display_table[col] = display_table[col].round(4)

    n_help = t("regional.col_n_help_balanced") if uses_balanced else t("regional.col_n_help_unbalanced")
    column_config = {
        "dialect_region": st.column_config.TextColumn(t("regional.col_region")),
        "DAT mean total cost": st.column_config.NumberColumn(t("regional.col_dat"), help=t("regional.col_dat_help")),
        "DIT mean total cost": st.column_config.NumberColumn(t("regional.col_dit"), help=t("regional.col_dit_help")),
        "total cost delta (DIT − DAT)": st.column_config.NumberColumn(t("regional.col_delta"),
                                                                       help=t("regional.col_delta_help")),
        "n sentences": st.column_config.NumberColumn(t("regional.col_n"), help=n_help),
    }

    st.dataframe(display_table, hide_index=True, width="stretch", column_config=column_config)
    chart_help(t("regional.summary_shows", subset=subset), t("common.table_use"))


def render_cost_distribution(per_sentence: pd.DataFrame, regions_sorted: list[str]) -> None:
    """Per-region/model box plot of per-sentence total alignment cost (5/25/50/75/95 percentiles)."""
    st.markdown(f"### {t('regional.distribution_title')}")

    box_stats = _box_stats(per_sentence)
    box_stats["model"] = _model_labels(box_stats["model"])
    base = alt.Chart(box_stats).encode(
        x=alt.X("dialect_region:N", sort=regions_sorted, title=None,
                axis=alt.Axis(labelAngle=0, labelOverlap=False)),
        color=alt.Color("model:N", scale=_model_scale(), legend=alt.Legend(title=None)),
        xOffset=alt.XOffset("model:N"),
    )
    whisker = base.mark_rule(strokeWidth=1.5).encode(
        y=alt.Y("min:Q", title=t("regional.distribution_axis")),
        y2="max:Q",
    )
    box = base.mark_bar(size=22, opacity=0.9).encode(y="q1:Q", y2="q3:Q")
    median_tick = base.mark_tick(thickness=2.5, color="black", size=22).encode(y="median:Q")

    st.altair_chart((whisker + box + median_tick).properties(height=380), width="stretch")
    chart_help(t("regional.distribution_shows"), t("regional.distribution_use"))


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
