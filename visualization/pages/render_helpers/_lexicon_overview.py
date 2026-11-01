"""Word-cloud overview for the Dialect Word Lexicon page."""
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import JsCode, st_echarts

from _data import REGION_COLORS, CloudMode, pair_region_counts, tfidf_matrix_pairs

_TABLE_COLUMNS = ["pair", "TF-IDF", "peak region", "count"]
_TOP_N = 200

_MODE_PAIR_LABEL: dict[CloudMode, str] = {
    "ref_dit": "(ref, DIT-hyp)",
    "dat_dit": "(DAT-hyp, DIT-hyp)",
}


def render_caption(mode: CloudMode = "ref_dit") -> None:
    """Section header and explanatory text above the word cloud."""
    pair_label = _MODE_PAIR_LABEL[mode]

    st.markdown("### Word Cloud for Dialect Specific Candidates")
    st.markdown(
        f"{pair_label} pairs ranked by their **TF-IDF** score across the selected regions (treated as documents). "
        "Surfaces pairs most distinct to their specific region. Hover a pair to see its score, peak region, and count."
    )


def compute_top_table(selected_regions: list[str], include_preterite: bool,
                      mode: CloudMode, dataset: str) -> pd.DataFrame:
    """Top-N pairs ranked by max TF-IDF across the selected regions, for the chosen mode."""
    matrix, vocab, _word_to_idx, region_order = tfidf_matrix_pairs(include_preterite, mode, dataset)

    selected_idx = [i for i, r in enumerate(region_order) if r in selected_regions]

    if not selected_idx:
        return pd.DataFrame(columns=_TABLE_COLUMNS)

    sub_matrix = matrix[selected_idx, :]
    scores = sub_matrix.max(axis=0)

    nonzero = np.flatnonzero(scores > 0)
    if nonzero.size == 0:
        return pd.DataFrame(columns=_TABLE_COLUMNS)

    order = nonzero[np.argsort(-scores[nonzero])][:_TOP_N]

    peak_region_indices = sub_matrix.argmax(axis=0)
    selected_region_names = [region_order[i] for i in selected_idx]

    counts_by_pair_region = pair_region_counts(dataset, tuple(selected_regions), include_preterite, mode)

    def peak_over_total(pair_str: str, peak_region: str) -> str:
        if pair_str not in counts_by_pair_region.index:
            return "0/0"
        row = counts_by_pair_region.loc[pair_str]
        return f"{int(row.get(peak_region, 0))}/{int(row.sum())}"

    return pd.DataFrame({
        "pair": [_decode_pair(vocab[i]) for i in order],
        "TF-IDF": scores[order],
        "peak region": [selected_region_names[int(peak_region_indices[i])] for i in order],
        "count": [
            peak_over_total(vocab[i], selected_region_names[int(peak_region_indices[i])])
            for i in order
        ],
    })


def render_word_cloud(table: pd.DataFrame) -> None:
    """Word cloud sized by TF-IDF, colored by peak region; hover shows score and count."""
    cloud_data = [
        {
            "name": str(row["pair"]),
            "value": float(row["TF-IDF"]),
            "peak": str(row["peak region"]),
            "count": str(row["count"]),
            "textStyle": {"color": REGION_COLORS.get(row["peak region"], "#333")},
        }
        for _, row in table.iterrows()
    ]
    option = {
        "tooltip": {
            "show": True,
            "formatter": JsCode(
                "function(p){return p.name + '<br/>TF-IDF: ' + p.value.toFixed(5) + "
                "'<br/>peak: ' + (p.data.peak || '—') + "
                "'<br/>count: ' + p.data.count;}"
            ),
        },
        "series": [{
            "type": "wordCloud",
            "shape": "circle",
            "left": "center",
            "top": "center",
            "width": "100%",
            "height": "100%",
            "sizeRange": [14, 80],
            "rotationRange": [-30, 30],
            "rotationStep": 15,
            "gridSize": 8,
            "drawOutOfBound": False,
            "emphasis": {
                "focus": "self",
                "textStyle": {"shadowBlur": 8, "shadowColor": "#999"},
            },
            "data": cloud_data,
        }],
    }
    st_echarts(options=option, height="520px")
    _render_region_legend(table)


def render_top_candidates_expander(table: pd.DataFrame, mode: CloudMode = "ref_dit") -> None:
    """Collapsible table listing the top regionally distinctive substitution pairs."""
    pair_label = _MODE_PAIR_LABEL[mode]
    pair_help = (
        "(ref, DIT-hyp) pair: Standard German reference word -> DIT (Whisper) hypothesis."
        if mode == "ref_dit"
        else "(DAT-hyp, DIT-hyp) pair: DAT (FHNW) hypothesis word -> DIT (Whisper) hypothesis."
    )
    column_config = {
        "pair": st.column_config.TextColumn(help=pair_help),
        "TF-IDF": st.column_config.NumberColumn(
            help=f"For each pair, the highest TF-IDF across the selected regions is shown. "
                 f"Documents = all 7 dialect regions; terms = {pair_label} pairs. "
                 "Higher = pair is more distinct to its specific region.",
            format="%.5f",
        ),
        "peak region": st.column_config.TextColumn(
            help="The selected region where this pair's TF-IDF is highest.",
        ),
        "count": st.column_config.TextColumn(
            help="Count in highest TF-IDF region / total count across the selected regions.",
        ),
    }
    with st.expander("Top Dialect Specific Candidates"):
        st.caption(
            f"{pair_label} pairs ranked by TF-IDF (max across selected regions). "
            "Surfaces pairs most distinct to their specific region."
        )
        st.dataframe(table, width="stretch", hide_index=True, column_config=column_config)
        st.caption("Default ordering: descending by **TF-IDF**.")


def _decode_pair(p: str) -> str:
    """'ref+hyp' (internal) → 'ref → hyp' (display)."""
    ref, hyp = p.split("+", 1)
    return f"{ref} → {hyp}"


def _render_region_legend(table: pd.DataFrame) -> None:
    """Inline color legend below the cloud, listing only regions that appear as peaks."""
    peaks_in_view = [r for r in REGION_COLORS if (table["peak region"] == r).any()]

    if not peaks_in_view:
        return

    swatches = " &nbsp; ".join(
        f'<span style="display:inline-block;width:12px;height:12px;background:{REGION_COLORS[r]};'
        f'margin-right:4px;vertical-align:middle;border-radius:2px;"></span>{r}'
        for r in peaks_in_view
    )
    st.markdown(
        f'<div style="text-align:center;font-size:0.9em;color:#333;">{swatches}</div>',
        unsafe_allow_html=True,
    )
