"""
Etch AI Decision Support System - Plotly 차트 빌더 모듈
모든 함수는 완성된 plotly Figure를 반환한다 (st.plotly_chart로 렌더링).

단위 참고: 실제 데이터셋 기준 CD/Depth는 nm, Uniformity 계열은 변동계수(CV%, 낮을수록 좋음).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from style import COLORS
from data_utils import PARTICLE_DEFECT_THRESHOLD, ZONE_ORDER

_FONT = dict(family="Pretendard, -apple-system, Segoe UI, sans-serif", color=COLORS["text_primary"])


def _status_colors_for_values(values, warn_th=90, good_th=95):
    """값이 높을수록 좋은 지표(Pass Rate 등)를 good/warning/critical 색상으로 변환"""
    colors = []
    for v in values:
        if v >= good_th:
            colors.append(COLORS["chart_good"])
        elif v >= warn_th:
            colors.append(COLORS["chart_warning"])
        else:
            colors.append(COLORS["chart_critical"])
    return colors


# ----------------------------------------------------------------------------
# 1. 공정 예측 탭 차트
# ----------------------------------------------------------------------------
def build_cd_bar_chart(result: dict):
    """Top / Mid / Bottom CD 막대그래프"""
    categories = ["Top CD", "Mid CD", "Bottom CD"]
    values = [result[c] for c in categories]
    colors = [COLORS["series1"], COLORS["series2"], COLORS["series3"]]

    fig = go.Figure(go.Bar(
        x=categories, y=values, marker_color=colors,
        text=[f"{v:.1f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title="CD 분포 (Top / Mid / Bottom)", yaxis_title="CD (nm)",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        margin=dict(t=50, b=30, l=30, r=30), showlegend=False, height=320,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"], zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def build_gauge_chart(title: str, value: float, warn_th: float = 90, good_th: float = 95):
    """값이 높을수록 좋은 지표용 게이지 (예: Overall Pass Rate)"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": "%", "font": {"size": 34}},
        title={"text": title, "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLORS["muted"]},
            "bar": {"color": COLORS["accent"]},
            "bgcolor": COLORS["surface"],
            "steps": [
                {"range": [0, warn_th], "color": "rgba(208,59,59,0.15)"},
                {"range": [warn_th, good_th], "color": "rgba(250,178,25,0.18)"},
                {"range": [good_th, 100], "color": "rgba(12,163,12,0.15)"},
            ],
            "threshold": {"line": {"color": COLORS["critical"], "width": 3}, "thickness": 0.8, "value": good_th},
        },
    ))
    fig.update_layout(paper_bgcolor=COLORS["surface"], font=_FONT, margin=dict(t=40, b=10, l=25, r=25), height=280)
    return fig


def build_variation_gauge(title: str, value: float, good_th: float, warn_th: float, max_range: float):
    """값이 낮을수록 좋은 지표용 게이지 (CD/Depth Uniformity·CV%)"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": "%", "font": {"size": 34}},
        title={"text": title, "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, max_range], "tickcolor": COLORS["muted"]},
            "bar": {"color": COLORS["accent"]},
            "bgcolor": COLORS["surface"],
            "steps": [
                {"range": [0, good_th], "color": "rgba(12,163,12,0.15)"},
                {"range": [good_th, warn_th], "color": "rgba(250,178,25,0.18)"},
                {"range": [warn_th, max_range], "color": "rgba(208,59,59,0.15)"},
            ],
            "threshold": {"line": {"color": COLORS["critical"], "width": 3}, "thickness": 0.8, "value": warn_th},
        },
    ))
    fig.update_layout(paper_bgcolor=COLORS["surface"], font=_FONT, margin=dict(t=40, b=10, l=25, r=25), height=280)
    return fig


# ----------------------------------------------------------------------------
# 2. Process Dashboard - 품질 결과 시각화 (Wafer 단위 추이, Wafer_Summary 원본 사용)
# ----------------------------------------------------------------------------
def build_cd_trend_chart(wafer_df):
    fig = go.Figure()
    cols = [("Top_CD_Mean_nm", "Top CD"), ("Mid_CD_Mean_nm", "Mid CD"), ("Bottom_CD_Mean_nm", "Bottom CD")]
    for (col, name), color in zip(cols, [COLORS["series1"], COLORS["series2"], COLORS["series3"]]):
        fig.add_trace(go.Scatter(
            x=wafer_df["Wafer_Label"], y=wafer_df[col], mode="lines+markers",
            name=name, line=dict(color=color, width=2), marker=dict(size=6),
        ))
    fig.update_layout(
        title="Wafer별 Top / Mid / Bottom CD", yaxis_title="CD (nm)",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        legend=dict(orientation="h", y=-0.32), margin=dict(t=50, b=60, l=30, r=20), height=340,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"])
    fig.update_xaxes(showgrid=False, tickangle=-45)
    return fig


def build_depth_trend_chart(wafer_df):
    fig = go.Figure(go.Bar(x=wafer_df["Wafer_Label"], y=wafer_df["Depth_Mean_nm"], marker_color=COLORS["series1"]))
    fig.update_layout(
        title="Wafer별 Depth", yaxis_title="Depth (nm)",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        margin=dict(t=50, b=60, l=30, r=20), height=340, showlegend=False,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"])
    fig.update_xaxes(showgrid=False, tickangle=-45)
    return fig


def build_cd_uniformity_trend_chart(wafer_df):
    cd_uniformity = wafer_df[["Top_CD_Uniformity_pct", "Mid_CD_Uniformity_pct", "Bottom_CD_Uniformity_pct"]].mean(axis=1)
    fig = go.Figure(go.Scatter(
        x=wafer_df["Wafer_Label"], y=cd_uniformity, mode="lines+markers",
        line=dict(color=COLORS["series1"], width=2), marker=dict(size=6),
    ))
    fig.update_layout(
        title="Wafer별 CD Uniformity (CV%, 낮을수록 좋음)", yaxis_title="CD Uniformity (CV%)",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        margin=dict(t=50, b=60, l=30, r=20), height=340, showlegend=False,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"])
    fig.update_xaxes(showgrid=False, tickangle=-45)
    return fig


def build_depth_uniformity_trend_chart(wafer_df):
    fig = go.Figure(go.Scatter(
        x=wafer_df["Wafer_Label"], y=wafer_df["Depth_Uniformity_pct"], mode="lines+markers",
        line=dict(color=COLORS["series2"], width=2), marker=dict(size=6),
    ))
    fig.update_layout(
        title="Wafer별 Depth Uniformity (CV%, 낮을수록 좋음)", yaxis_title="Depth Uniformity (CV%)",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        margin=dict(t=50, b=60, l=30, r=20), height=340, showlegend=False,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"])
    fig.update_xaxes(showgrid=False, tickangle=-45)
    return fig


def build_pass_rate_trend_chart(wafer_df):
    colors = _status_colors_for_values(wafer_df["Overall_Spec_Pass_Rate_pct"])
    fig = go.Figure(go.Bar(x=wafer_df["Wafer_Label"], y=wafer_df["Overall_Spec_Pass_Rate_pct"], marker_color=colors))
    fig.add_hline(y=95, line_dash="dash", line_color=COLORS["good"], annotation_text="양호 ≥95%")
    fig.add_hline(y=90, line_dash="dash", line_color="#c98500", annotation_text="주의 ≥90%")
    fig.update_layout(
        title="Wafer별 Overall Pass Rate", yaxis_title="Pass Rate (%)",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        margin=dict(t=50, b=60, l=30, r=20), height=340, showlegend=False,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"], range=[0, 100])
    fig.update_xaxes(showgrid=False, tickangle=-45)
    return fig


def build_particle_chart(wafer_df):
    has_defect = wafer_df["Total_Defect_Count"] > PARTICLE_DEFECT_THRESHOLD
    counts = has_defect.value_counts()
    labels = ["발생" if v else "미발생" for v in counts.index]
    colors = [COLORS["chart_critical"] if v else COLORS["chart_good"] for v in counts.index]
    fig = go.Figure(go.Bar(
        x=labels, y=counts.values, marker_color=colors,
        text=counts.values, textposition="outside",
    ))
    fig.update_layout(
        title=f"Particle(Defect &gt;{PARTICLE_DEFECT_THRESHOLD}건) 발생 Wafer 수", yaxis_title="Wafer 수",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        margin=dict(t=50, b=30, l=30, r=20), height=340, showlegend=False,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"])
    return fig


# ----------------------------------------------------------------------------
# 2-1. Process Dashboard - Rev별 품질 변화 시각화
# -----------------------------------------------------------------------------
def build_rev_cd_trend_chart(rev_df):
    """Recipe(Rev)별 Top/Mid/Bottom CD 실측 평균을 비교한다."""
    fig = go.Figure()
    columns = [("Top CD", "Top CD"), ("Mid CD", "Mid CD"), ("Bottom CD", "Bottom CD")]
    for (column, name), color in zip(columns, [COLORS["series1"], COLORS["series2"], COLORS["series3"]]):
        fig.add_trace(go.Scatter(
            x=rev_df["Recipe"], y=rev_df[column], mode="lines+markers",
            name=name, line=dict(color=color, width=2), marker=dict(size=7),
            customdata=rev_df["Wafer 수"],
            hovertemplate=f"%{{x}}<br>{name}: %{{y:.1f}} nm<br>Wafer 수: %{{customdata}}장<extra></extra>",
        ))
    fig.update_layout(
        title="Rev별 Top / Mid / Bottom CD 평균", yaxis_title="CD (nm)",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        legend=dict(orientation="h", y=-0.28), margin=dict(t=50, b=50, l=30, r=20), height=340,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"])
    fig.update_xaxes(showgrid=False, type="category", tickangle=-30)
    return fig


def build_rev_depth_trend_chart(rev_df):
    """Recipe(Rev)별 Depth 실측 평균을 비교한다."""
    fig = go.Figure(go.Bar(
        x=rev_df["Recipe"], y=rev_df["Depth"], marker_color=COLORS["series1"],
        customdata=rev_df["Wafer 수"],
        hovertemplate="%{x}<br>Depth: %{y:.1f} nm<br>Wafer 수: %{customdata}장<extra></extra>",
    ))
    fig.update_layout(
        title="Rev별 Depth 평균", yaxis_title="Depth (nm)",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        margin=dict(t=50, b=50, l=30, r=20), height=340, showlegend=False, bargap=0.35,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"])
    fig.update_xaxes(showgrid=False, type="category", tickangle=-30)
    return fig


def build_rev_uniformity_trend_chart(rev_df):
    """Recipe(Rev)별 CD/Depth Uniformity 실측 평균을 함께 비교한다."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rev_df["Recipe"], y=rev_df["CD Uniformity"], mode="lines+markers",
        name="CD Uniformity", line=dict(color=COLORS["series1"], width=2), marker=dict(size=7),
        customdata=rev_df["Wafer 수"],
        hovertemplate="%{x}<br>CD Uniformity: %{y:.2f}%<br>Wafer 수: %{customdata}장<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=rev_df["Recipe"], y=rev_df["Depth Uniformity"], mode="lines+markers",
        name="Depth Uniformity", line=dict(color=COLORS["series2"], width=2), marker=dict(size=7),
        customdata=rev_df["Wafer 수"],
        hovertemplate="%{x}<br>Depth Uniformity: %{y:.2f}%<br>Wafer 수: %{customdata}장<extra></extra>",
    ))
    fig.update_layout(
        title="Rev별 Uniformity 비교 (CV%, 낮을수록 좋음)", yaxis_title="Uniformity (CV%)",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        legend=dict(orientation="h", y=-0.28), margin=dict(t=50, b=50, l=30, r=20), height=340,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"])
    fig.update_xaxes(showgrid=False, type="category", tickangle=-30)
    return fig


def build_rev_pass_rate_chart(rev_df):
    """Recipe(Rev)별 Overall Spec Pass Rate 실측 평균을 비교한다."""
    colors = _status_colors_for_values(rev_df["Overall Spec Pass Rate"])
    fig = go.Figure(go.Bar(
        x=rev_df["Recipe"], y=rev_df["Overall Spec Pass Rate"], marker_color=colors,
        customdata=rev_df["Wafer 수"],
        hovertemplate="%{x}<br>Pass Rate: %{y:.1f}%<br>Wafer 수: %{customdata}장<extra></extra>",
    ))
    fig.add_hline(
        y=95, line_dash="dash", line_color=COLORS["chart_good"], line_width=2,
        annotation_text="양호 ≥95%", annotation_position="top left",
    )
    fig.add_hline(
        y=90, line_dash="dash", line_color=COLORS["chart_warning"], line_width=2,
        annotation_text="주의 ≥90%", annotation_position="bottom left",
    )
    fig.update_layout(
        title="Rev별 Overall Pass Rate", yaxis_title="Pass Rate (%)",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        margin=dict(t=50, b=50, l=30, r=20), height=340, showlegend=False, bargap=0.35,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"], range=[0, 100])
    fig.update_xaxes(showgrid=False, type="category", tickangle=-30)
    return fig


def build_rev_defect_chart(rev_defect_df):
    """Recipe(Rev)별 Particle 기준 초과 Wafer 수와 평균 Defect Count를 보여준다."""
    colors = [
        COLORS["chart_critical"] if value > 0 else COLORS["chart_good"]
        for value in rev_defect_df["Particle 발생 Wafer 수"]
    ]
    fig = go.Figure(go.Bar(
        x=rev_defect_df["Recipe"], y=rev_defect_df["Particle 발생 Wafer 수"], marker_color=colors,
        customdata=np.stack([rev_defect_df["Wafer 수"], rev_defect_df["평균 Defect Count"]], axis=-1),
        hovertemplate=(
            "%{x}<br>Particle 발생 Wafer 수: %{y} / %{customdata[0]}장"
            "<br>평균 Defect Count: %{customdata[1]:.1f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=f"Rev별 Particle(Defect >{PARTICLE_DEFECT_THRESHOLD}건) 발생 Wafer 수", yaxis_title="Wafer 수",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        margin=dict(t=50, b=45, l=30, r=20), height=340, showlegend=False, bargap=0.35,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"], rangemode="tozero", dtick=1)
    fig.update_xaxes(showgrid=False, type="category", tickangle=-30)
    return fig


# ----------------------------------------------------------------------------
# 3. Wafer Map (Site_Level_Raw의 실제 Radius_frac/Angle_deg 좌표 사용)
# ----------------------------------------------------------------------------
def build_wafer_map(filtered_site_df, value_col: str, value_label: str):
    """선택 조건에 해당하는 모든 Wafer의 같은 Site 위치값을 평균해 대표 Wafer Map을 그린다."""
    agg = (
        filtered_site_df.groupby(["Site_ID", "Zone", "Radius_frac", "Angle_deg"], dropna=False)[value_col]
        .mean()
        .reset_index()
    )

    xs, ys, vals, zone_labels = [], [], [], []
    for _, row in agg.iterrows():
        angle = row["Angle_deg"]
        theta = 0.0 if pd.isna(angle) else np.radians(angle)
        radius = row["Radius_frac"]
        xs.append(radius * np.cos(theta))
        ys.append(radius * np.sin(theta))
        vals.append(row[value_col])
        zone_labels.append(row["Zone"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(
            size=20, color=vals, colorscale=COLORS["wafer_colorscale"], showscale=True,
            colorbar=dict(title=value_label, thickness=14),
            line=dict(width=0.5, color=COLORS["surface"]),
        ),
        text=[f"{z}<br>{value_label}: {v:.2f}" for z, v in zip(zone_labels, vals)],
        hoverinfo="text", showlegend=False,
    ))
    theta_full = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(
        x=np.cos(theta_full), y=np.sin(theta_full), mode="lines",
        line=dict(color=COLORS["muted"], width=1.5), hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        title=f"Wafer Map — {value_label} (선택 조건 평균)",
        xaxis=dict(visible=False, range=[-1.15, 1.15], scaleanchor="y"),
        yaxis=dict(visible=False, range=[-1.15, 1.15]),
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        margin=dict(t=50, b=10, l=10, r=10), height=460,
    )
    return fig


_ZONE_SHORT_LABEL = {"Center": "Center", "Mid": "Mid", "Edge": "Edge", "Extreme Edge": "ExtEdge"}
_ZONE_LABEL_COLOR = {
    "Center": COLORS["chart_critical"],
    "Mid": COLORS["series1"],
    "Edge": COLORS["series3"],
    "Extreme Edge": COLORS["series2"],
}
_ZONE_PROFILE_BG = {
    "Center": "rgba(199,131,131,0.09)",
    "Mid": "rgba(120,153,184,0.08)",
    "Edge": "rgba(167,153,196,0.08)",
    "Extreme Edge": "rgba(115,173,159,0.08)",
}


def build_wafer_profile_chart(filtered_site_df, value_col: str, value_label: str):
    """Edge → Center → Edge 단면을 Zone 배경·구간명·희소 tick과 함께 표시한다.

    Site 집계와 Point 배열 방식은 기존과 같고, 가독성을 위한 표시만 강화한다.
    """
    agg = (
        filtered_site_df.groupby(["Site_ID", "Zone", "Radius_frac", "Angle_deg"], dropna=False)[value_col]
        .mean()
        .reset_index()
    )
    zone_rank = {zone: index for index, zone in enumerate(ZONE_ORDER)}
    agg["_zone_rank"] = agg["Zone"].map(zone_rank).fillna(len(ZONE_ORDER))
    agg["_angle_sort"] = agg["Angle_deg"].fillna(-1)

    center = agg[agg["Zone"] == "Center"].sort_values("_angle_sort")
    outer = agg[agg["Zone"] != "Center"].sort_values(
        ["_zone_rank", "_angle_sort"], ascending=[False, True]
    )
    ordered = pd.concat([outer, center, outer.iloc[::-1]], ignore_index=True)
    ordered["Point"] = range(1, len(ordered) + 1)

    title = (
        f"{value_label} — Wafer 단면 Profile "
        f"<span style='font-size:0.7em;color:{COLORS['muted']}'>(ExtEdge → Center → ExtEdge)</span>"
    )
    fig = go.Figure()
    if ordered.empty:
        fig.add_annotation(
            text="단면 Profile을 그릴 Site 데이터가 부족합니다.",
            showarrow=False,
            font=dict(size=12, color=COLORS["muted"]),
        )
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(t=55, b=40, l=30, r=20), height=380,
        )
        return fig

    run_id = (ordered["Zone"] != ordered["Zone"].shift()).cumsum()
    zone_runs = (
        ordered.groupby(run_id)
        .agg(Zone=("Zone", "first"), Start=("Point", "min"), End=("Point", "max"))
        .reset_index(drop=True)
    )

    for _, run in zone_runs.iterrows():
        background = _ZONE_PROFILE_BG.get(run["Zone"])
        if background:
            fig.add_vrect(
                x0=run["Start"] - 0.5,
                x1=run["End"] + 0.5,
                fillcolor=background,
                line_width=0,
                layer="below",
            )
    for index in range(len(zone_runs) - 1):
        fig.add_vline(
            x=zone_runs.loc[index, "End"] + 0.5,
            line_width=1,
            line_color=COLORS["gridline"],
        )

    for _, run in zone_runs.iterrows():
        midpoint = (run["Start"] + run["End"]) / 2
        is_center = run["Zone"] == "Center"
        fig.add_annotation(
            x=midpoint,
            xref="x",
            y=0.94 if is_center else 1.02,
            yref="paper",
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            text=f"<b>{_ZONE_SHORT_LABEL.get(run['Zone'], run['Zone'])}</b>",
            font=dict(size=10, color=_ZONE_LABEL_COLOR.get(run["Zone"], COLORS["muted"])),
            bgcolor="rgba(255,255,255,0.78)" if is_center else None,
        )

    point_labels = ordered["Point"].map(lambda point: f"P{point:02d}")
    angle_text = ordered["Angle_deg"].map(lambda angle: "—" if pd.isna(angle) else f"{angle:.0f}°")
    marker_colors = [
        COLORS["chart_critical"] if zone == "Center" else COLORS["series1"]
        for zone in ordered["Zone"]
    ]
    fig.add_trace(go.Scatter(
        x=ordered["Point"], y=ordered[value_col], mode="lines+markers",
        line=dict(color=COLORS["series1"], width=2),
        marker=dict(size=6, color=marker_colors),
        customdata=np.stack([
            point_labels,
            ordered["Site_ID"].astype(str),
            ordered["Zone"],
            ordered["Radius_frac"].round(2).astype(str),
            angle_text,
        ], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
            "Zone: %{customdata[2]}<br>"
            f"{value_label}: " + "%{y:.2f} nm<br>"
            "Radius: %{customdata[3]}<br>"
            "Angle: %{customdata[4]}<extra></extra>"
        ),
        showlegend=False,
    ))

    if not center.empty:
        center_point = float(ordered.loc[ordered["Zone"] == "Center", "Point"].mean())
        fig.add_vline(
            x=center_point,
            line_dash="dash",
            line_width=2,
            line_color=COLORS["chart_critical"],
        )

    point_count = len(ordered)
    tick_values = sorted({int(round(value)) for value in np.linspace(1, point_count, num=min(5, point_count))})
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Wafer 단면 위치 (측정 Point 순서)",
        yaxis_title=f"{value_label} (nm)",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        margin=dict(t=85, b=55, l=30, r=20), height=400, showlegend=False,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"])
    fig.update_xaxes(
        showgrid=False,
        range=[0.3, point_count + 0.7],
        tickmode="array",
        tickvals=tick_values,
        ticktext=[str(value) for value in tick_values],
        tickfont=dict(size=11),
    )
    return fig


def build_recipe_score_chart(scoreboard_df: pd.DataFrame):
    """Recipe(Rev)별 종합 품질 점수 막대그래프 — 실측 데이터 기반, 점수 높은 순 정렬."""
    colors = _status_colors_for_values(scoreboard_df["종합 점수"], warn_th=50, good_th=80)
    fig = go.Figure(go.Bar(
        x=scoreboard_df["Recipe"], y=scoreboard_df["종합 점수"], marker_color=colors,
        text=[f"{v:.1f}" for v in scoreboard_df["종합 점수"]], textposition="outside",
    ))
    fig.update_layout(
        title="Recipe(Rev)별 종합 품질 점수 (실측 기반)", yaxis_title="종합 점수",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        margin=dict(t=50, b=60, l=30, r=20), height=360, showlegend=False,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"], range=[0, 105])
    fig.update_xaxes(showgrid=False, tickangle=-45)
    return fig


# ----------------------------------------------------------------------------
# 4. Zone 분석 (Site_Level_Raw 기반 zone_summary)
# ----------------------------------------------------------------------------
def build_zone_cd_chart(zone_summary):
    fig = go.Figure()
    for col, color in zip(["Top CD", "Mid CD", "Bottom CD"],
                           [COLORS["series1"], COLORS["series2"], COLORS["series3"]]):
        fig.add_trace(go.Bar(x=zone_summary["Zone"], y=zone_summary[col], name=col, marker_color=color))
    fig.update_layout(
        title="Zone별 Top / Mid / Bottom CD", yaxis_title="CD (nm)", barmode="group",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        legend=dict(orientation="h", y=-0.22), margin=dict(t=50, b=40, l=30, r=20), height=340,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"])
    return fig


def build_zone_depth_chart(zone_summary):
    fig = go.Figure(go.Bar(x=zone_summary["Zone"], y=zone_summary["Depth"], marker_color=COLORS["series1"]))
    fig.update_layout(
        title="Zone별 Depth", yaxis_title="Depth (nm)",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        margin=dict(t=50, b=30, l=30, r=20), height=340, showlegend=False,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"])
    return fig


def build_zone_spread_chart(zone_summary):
    """Zone별 Uniformity 대체 지표: 같은 Zone 내 Site간 편차(표준편차, nm)"""
    fig = go.Figure()
    for col, color in zip(["CD Spread", "Depth Spread"], [COLORS["series1"], COLORS["series2"]]):
        fig.add_trace(go.Bar(x=zone_summary["Zone"], y=zone_summary[col], name=col, marker_color=color))
    fig.update_layout(
        title="Zone별 Site간 편차 (CD/Depth Spread, 낮을수록 균일)", yaxis_title="표준편차 (nm)", barmode="group",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        legend=dict(orientation="h", y=-0.22), margin=dict(t=50, b=40, l=30, r=20), height=340,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"])
    return fig


def build_zone_pass_rate_chart(zone_summary):
    colors = _status_colors_for_values(zone_summary["Pass Rate"])
    fig = go.Figure(go.Bar(x=zone_summary["Zone"], y=zone_summary["Pass Rate"], marker_color=colors))
    fig.update_layout(
        title="Zone별 Overall Pass Rate", yaxis_title="Pass Rate (%)",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        margin=dict(t=50, b=30, l=30, r=20), height=340, showlegend=False,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"], range=[0, 100])
    return fig


def build_zone_defect_chart(zone_summary):
    fig = go.Figure(go.Bar(
        x=zone_summary["Zone"], y=zone_summary["Defect Rate"], marker_color=COLORS["chart_critical"],
    ))
    fig.update_layout(
        title="Zone별 Defect(Particle) 발생률", yaxis_title="Defect 발생률 (%)",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        margin=dict(t=50, b=30, l=30, r=20), height=340, showlegend=False,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"])
    return fig


# ----------------------------------------------------------------------------
# 5. 현재 Recipe vs 추천 Recipe 비교 (Output D)
# ----------------------------------------------------------------------------
def build_cd_comparison_chart(current_result: dict, recommended_result: dict):
    """Top/Mid/Bottom CD를 현재 vs 추천 Recipe로 나란히 비교"""
    categories = ["Top CD", "Mid CD", "Bottom CD"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories, y=[current_result[c] for c in categories],
        name="현재 Recipe", marker_color=COLORS["muted"],
    ))
    fig.add_trace(go.Bar(
        x=categories, y=[recommended_result[c] for c in categories],
        name="추천 Recipe", marker_color=COLORS["accent"],
    ))
    fig.update_layout(
        title="CD 비교 (현재 vs 추천)", yaxis_title="CD (nm)", barmode="group",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=40, l=30, r=20), height=340,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"])
    return fig


def build_score_comparison_chart(current_score: float, recommended_score: float):
    """종합 품질 점수를 현재 vs 추천 Recipe로 비교"""
    labels = ["현재 Recipe", "추천 Recipe"]
    values = [current_score, recommended_score]
    colors = [COLORS["muted"], COLORS["accent"]]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.1f}점" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title="종합 품질 점수 비교", yaxis_title="점수",
        plot_bgcolor=COLORS["surface"], paper_bgcolor=COLORS["surface"], font=_FONT,
        margin=dict(t=50, b=30, l=30, r=20), height=340, showlegend=False,
    )
    fig.update_yaxes(gridcolor=COLORS["gridline"], range=[0, 100])
    return fig
