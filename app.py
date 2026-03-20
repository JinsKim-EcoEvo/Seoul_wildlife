import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
from pyproj import Transformer
from sklearn.cluster import DBSCAN
import plotly.express as px

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="Seoul Biodiversity Explorer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

GITHUB_RAW_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "JinsKim-EcoEvo/Seoul_wildlife/"
    "f3f7e06d025e776da585e9960ecb38906ffe3e86/"
    "Seoul_wildlife.csv"
)

DEFAULT_SOURCE_CRS = "EPSG:2097"
SEOUL_CENTER = [37.5665, 126.9780]

# =========================================================
# 스타일
# =========================================================
st.markdown(
    """
    <style>
    :root {
        --bg: #f4f7f5;
        --card: #ffffff;
        --border: #dfe7e2;
        --text: #1f2d26;
        --muted: #66756d;
        --accent: #1f6b4f;
        --accent-soft: #e9f5ef;
        --accent-2: #2f8f6b;
    }

    .stApp {
        background: linear-gradient(180deg, #f7faf8 0%, #f1f5f2 100%);
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }

    .hero-wrap {
        background: linear-gradient(135deg, #1f6b4f 0%, #245f86 100%);
        border-radius: 22px;
        padding: 28px 30px 24px 30px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(20, 50, 35, 0.12);
    }

    .hero-title {
        font-size: 2.0rem;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 0.35rem;
    }

    .hero-sub {
        font-size: 1rem;
        line-height: 1.6;
        color: rgba(255,255,255,0.92);
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--text);
        margin-top: 0.1rem;
        margin-bottom: 0.8rem;
    }

    .panel-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 4px 14px rgba(35, 55, 45, 0.05);
        margin-bottom: 14px;
    }

    .soft-card {
        background: #f8fbf9;
        border: 1px solid #e5ece8;
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }

    .small-label {
        font-size: 0.82rem;
        color: var(--muted);
        margin-bottom: 0.15rem;
    }

    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 4px 14px rgba(35, 55, 45, 0.05);
    }

    .metric-title {
        font-size: 0.84rem;
        color: var(--muted);
        margin-bottom: 0.25rem;
    }

    .metric-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: var(--text);
    }

    .metric-sub {
        font-size: 0.82rem;
        color: var(--accent);
        margin-top: 0.2rem;
    }

    .legend-box {
        position: relative;
        z-index: 9999;
        background: rgba(255,255,255,0.96);
        border: 1px solid #d8e3dc;
        border-radius: 12px;
        padding: 10px 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        font-size: 12px;
        line-height: 1.5;
    }

    .legend-title {
        font-weight: 700;
        margin-bottom: 6px;
        color: #22332b;
    }

    .legend-item {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
        color: #31453b;
    }

    .legend-dot {
        width: 11px;
        height: 11px;
        border-radius: 50%;
        margin-right: 8px;
        border: 1px solid rgba(0,0,0,0.15);
        flex-shrink: 0;
    }

    .sidebar-title {
        font-size: 1rem;
        font-weight: 800;
        color: #23352c;
        margin-bottom: 0.8rem;
    }

    .sidebar-note {
        font-size: 0.86rem;
        color: #617168;
        line-height: 1.55;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7fbf8 0%, #f0f5f2 100%);
        border-right: 1px solid #e2ebe5;
    }

    div[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    .footer-note {
        font-size: 0.85rem;
        color: #64746d;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 유틸
# =========================================================
def safe_read_csv(url: str) -> pd.DataFrame:
    last_error = None
    for enc in ["cp949", "utf-8-sig", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(url, encoding=enc, low_memory=False)
        except Exception as e:
            last_error = e
    raise last_error


def clean_text_column(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype="object")
    return (
        df[col]
        .astype(str)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )


def parse_year(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def convert_to_wgs84(
    df: pd.DataFrame,
    x_col: str = "X좌표",
    y_col: str = "Y좌표",
    source_crs: str = DEFAULT_SOURCE_CRS,
) -> pd.DataFrame:
    out = df.copy()
    out[x_col] = pd.to_numeric(out[x_col], errors="coerce")
    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")

    valid = out[[x_col, y_col]].dropna().copy()
    if valid.empty:
        out["lon"] = np.nan
        out["lat"] = np.nan
        return out

    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(valid[x_col].values, valid[y_col].values)

    out["lon"] = np.nan
    out["lat"] = np.nan
    out.loc[valid.index, "lon"] = lon
    out.loc[valid.index, "lat"] = lat
    return out


def filter_to_seoul_extent(df: pd.DataFrame) -> pd.DataFrame:
    cond = (
        df["lat"].between(37.35, 37.75, inclusive="both")
        & df["lon"].between(126.70, 127.25, inclusive="both")
    )
    return df.loc[cond].copy()


def run_dbscan(df: pd.DataFrame, eps_meters: int, min_samples: int) -> pd.DataFrame:
    work = df.dropna(subset=["lat", "lon"]).copy()
    if work.empty:
        work["cluster"] = np.nan
        return work

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    mx, my = transformer.transform(work["lon"].values, work["lat"].values)
    coords = np.column_stack([mx, my])

    model = DBSCAN(eps=eps_meters, min_samples=min_samples)
    labels = model.fit_predict(coords)
    work["cluster"] = labels
    return work


def make_popup_html(row: pd.Series) -> str:
    def v(key):
        value = row.get(key, "-")
        return "-" if pd.isna(value) else value

    return f"""
    <div style="font-size:13px; line-height:1.55; color:#22332b;">
        <b>국명</b>: {v('국명')}<br>
        <b>학명</b>: <i>{v('학명')}</i><br>
        <b>출현년도</b>: {v('출현년도')}<br>
        <b>서식지명</b>: {v('서식지명')}<br>
        <b>세부지역</b>: {v('세부통계용명칭')}<br>
        <b>원전</b>: {v('원전')}
    </div>
    """


def build_base_map(center=None, zoom_start=11):
    if center is None:
        center = SEOUL_CENTER
    return folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles="CartoDB positron",
        control_scale=True,
    )


def add_occurrence_markers(m: folium.Map, df: pd.DataFrame, sample_n: int = 1800):
    points = df.dropna(subset=["lat", "lon"]).copy()
    if points.empty:
        return

    if len(points) > sample_n:
        points = points.sample(sample_n, random_state=42)

    marker_cluster = MarkerCluster(name="출현 지점").add_to(m)

    for _, row in points.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color="#2f6fdd",
            weight=1,
            fill=True,
            fill_color="#4d8df7",
            fill_opacity=0.72,
            popup=folium.Popup(make_popup_html(row), max_width=320),
            tooltip=row.get("국명", "출현 지점"),
        ).add_to(marker_cluster)


def add_heatmap(m: folium.Map, df: pd.DataFrame):
    points = df.dropna(subset=["lat", "lon"])
    if points.empty:
        return
    heat_data = points[["lat", "lon"]].values.tolist()
    HeatMap(
        heat_data,
        name="Hotspot Heatmap",
        radius=20,
        blur=16,
        min_opacity=0.30,
    ).add_to(m)


def add_cluster_markers(m: folium.Map, df: pd.DataFrame):
    clustered = df.dropna(subset=["lat", "lon", "cluster"]).copy()
    clustered = clustered[clustered["cluster"] != -1]
    if clustered.empty:
        return

    cluster_summary = (
        clustered.groupby("cluster")
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            count=("cluster", "size"),
            species_n=("학명", "nunique"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )

    layer = folium.FeatureGroup(name="클러스터 중심").add_to(m)

    for _, row in cluster_summary.iterrows():
        popup = f"""
        <div style="font-size:13px; line-height:1.55; color:#22332b;">
            <b>클러스터 ID</b>: {int(row['cluster'])}<br>
            <b>출현 기록 수</b>: {int(row['count'])}<br>
            <b>고유 학명 수</b>: {int(row['species_n'])}
        </div>
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=9,
            color="#d9485f",
            weight=2,
            fill=True,
            fill_color="#eb6a7d",
            fill_opacity=0.95,
            popup=folium.Popup(popup, max_width=260),
            tooltip=f"Cluster {int(row['cluster'])}",
        ).add_to(layer)


def add_custom_legend(m: folium.Map):
    legend_html = """
    <div style="
        position: fixed;
        bottom: 26px;
        left: 26px;
        z-index: 999999;
        ">
        <div class="legend-box">
            <div class="legend-title">지도 범례</div>
            <div class="legend-item">
                <span class="legend-dot" style="background:#4d8df7;"></span>
                종 출현 지점
            </div>
            <div class="legend-item">
                <span class="legend-dot" style="background:#eb6a7d;"></span>
                클러스터 중심
            </div>
            <div class="legend-item">
                <span style="
                    width: 16px;
                    height: 10px;
                    display: inline-block;
                    margin-right: 8px;
                    border-radius: 10px;
                    background: linear-gradient(90deg, #3b82f6 0%, #fde047 50%, #ef4444 100%);
                    border: 1px solid rgba(0,0,0,0.12);
                "></span>
                Hotspot Heatmap
            </div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


@st.cache_data(show_spinner=False)
def load_and_prepare_data(url: str, source_crs: str) -> pd.DataFrame:
    df = safe_read_csv(url)

    expected_cols = [
        "종코드", "국명", "학명", "서식지코드", "서식지명",
        "세부통계용명칭", "출현년도", "원전", "X좌표", "Y좌표", "서식지비고정보"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    text_cols = [
        "종코드", "국명", "학명", "서식지코드", "서식지명",
        "세부통계용명칭", "원전", "서식지비고정보"
    ]
    for col in text_cols:
        df[col] = clean_text_column(df, col)

    df["출현년도"] = parse_year(df["출현년도"])
    df = convert_to_wgs84(df, source_crs=source_crs)
    df = filter_to_seoul_extent(df)
    return df


def make_bar_species_by_region(df: pd.DataFrame):
    chart_df = (
        df.dropna(subset=["세부통계용명칭", "학명"])
        .groupby("세부통계용명칭")["학명"]
        .nunique()
        .reset_index(name="종수")
        .sort_values("종수", ascending=False)
        .head(15)
    )
    if chart_df.empty:
        return None
    fig = px.bar(
        chart_df,
        x="세부통계용명칭",
        y="종수",
        title="상위 지역별 고유 종수",
        text_auto=True,
    )
    fig.update_layout(
        xaxis_title="지역",
        yaxis_title="고유 종수",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def make_line_yearly(df: pd.DataFrame):
    chart_df = (
        df.dropna(subset=["출현년도"])
        .groupby("출현년도")
        .size()
        .reset_index(name="출현기록수")
        .sort_values("출현년도")
    )
    if chart_df.empty:
        return None
    fig = px.line(
        chart_df,
        x="출현년도",
        y="출현기록수",
        markers=True,
        title="연도별 출현 기록 수"
    )
    fig.update_layout(
        xaxis_title="출현년도",
        yaxis_title="출현 기록 수",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


def make_top_species_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.dropna(subset=["학명"])
        .groupby(["국명", "학명"])
        .size()
        .reset_index(name="출현기록수")
        .sort_values("출현기록수", ascending=False)
        .head(10)
    )


def get_cluster_detail(df: pd.DataFrame, cluster_id):
    if cluster_id is None:
        return None

    cluster_df = df[df["cluster"] == cluster_id].copy()
    if cluster_df.empty:
        return None

    species_table = (
        cluster_df.groupby(["국명", "학명"])
        .size()
        .reset_index(name="빈도")
        .sort_values("빈도", ascending=False)
        .head(15)
    )

    habitat_mode = "-"
    if cluster_df["서식지명"].dropna().shape[0] > 0:
        habitat_mode = cluster_df["서식지명"].mode().iloc[0]

    region_mode = "-"
    if cluster_df["세부통계용명칭"].dropna().shape[0] > 0:
        region_mode = cluster_df["세부통계용명칭"].mode().iloc[0]

    return {
        "cluster_id": cluster_id,
        "records": len(cluster_df),
        "species_n": cluster_df["학명"].nunique(),
        "year_min": cluster_df["출현년도"].dropna().min() if cluster_df["출현년도"].notna().any() else "-",
        "year_max": cluster_df["출현년도"].dropna().max() if cluster_df["출현년도"].notna().any() else "-",
        "대표서식지": habitat_mode,
        "대표지역": region_mode,
        "species_table": species_table,
    }


def metric_card(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 헤더
# =========================================================
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-title">Seoul Biodiversity Explorer</div>
        <div class="hero-sub">
            서울시 야생동식물 출현자료를 기반으로 공간적 집중지역과 생물다양성 패턴을 탐색하는
            인터랙티브 분석 플랫폼입니다. 연구자, 정책 담당자, 그리고 도시생태에 관심 있는 사용자들이
            직관적으로 데이터를 해석할 수 있도록 설계했습니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.radio(
    "페이지 선택",
    ["대시보드", "분석", "데이터 정보", "소개"],
    horizontal=True,
    label_visibility="collapsed",
)

# =========================================================
# 사이드바
# =========================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">탐색 필터</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-note">필터를 조정해 출현 패턴, hotspot, 그리고 공간 클러스터 구조를 탐색할 수 있습니다.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    with st.expander("기본 설정", expanded=True):
        source_crs = st.selectbox(
            "원본 좌표계",
            options=["EPSG:2097", "EPSG:5174", "EPSG:5181", "EPSG:4326"],
            index=0,
            help="점 위치가 서울 밖으로 보이면 다른 좌표계를 시도해보세요.",
        )

    try:
        df = load_and_prepare_data(GITHUB_RAW_CSV_URL, source_crs)
    except Exception as e:
        st.error("CSV를 불러오지 못했습니다.")
        st.exception(e)
        st.stop()

    if df.empty:
        st.warning("불러온 데이터가 비어 있습니다.")
        st.stop()

    all_years = sorted([int(y) for y in df["출현년도"].dropna().unique().tolist()])
    all_species = sorted(df["국명"].dropna().unique().tolist())
    all_habitats = sorted(df["서식지명"].dropna().unique().tolist())

    with st.expander("데이터 필터", expanded=True):
        if all_years:
            year_range = st.slider(
                "출현년도 범위",
                min_value=min(all_years),
                max_value=max(all_years),
                value=(min(all_years), max(all_years)),
            )
        else:
            year_range = None
            st.info("출현년도 정보가 없어 연도 필터를 사용할 수 없습니다.")

        selected_species = st.multiselect(
            "종 선택 (국명)",
            options=all_species,
            placeholder="전체 종",
        )

        selected_habitats = st.multiselect(
            "서식지 유형",
            options=all_habitats,
            placeholder="전체 서식지",
        )

    with st.expander("시각화 옵션", expanded=True):
        show_points = st.toggle("출현 지점 표시", value=True)
        show_heatmap = st.toggle("Hotspot Heatmap 표시", value=True)
        show_cluster_centers = st.toggle("클러스터 중심 표시", value=True)
        map_height = st.slider("지도 높이", min_value=520, max_value=980, value=690, step=10)

    with st.expander("클러스터링 설정", expanded=False):
        eps_meters = st.slider(
            "거리 기준 eps (m)",
            min_value=100,
            max_value=3000,
            value=700,
            step=100,
            help="값이 클수록 더 넓은 범위를 하나의 클러스터로 묶습니다.",
        )
        min_samples = st.slider(
            "최소 샘플 수",
            min_value=3,
            max_value=50,
            value=10,
            step=1,
            help="클러스터로 인정하기 위한 최소 출현기록 수입니다.",
        )

    st.markdown("---")
    st.markdown(
        """
        <div class="footer-note">
            해석 주의: 이 데이터는 문헌 기반 출현자료이므로 조사 강도 차이와 표본 편향이 반영될 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# 필터 적용
# =========================================================
filtered = df.copy()

if year_range is not None:
    filtered = filtered[
        filtered["출현년도"].between(year_range[0], year_range[1], inclusive="both")
    ]

if selected_species:
    filtered = filtered[filtered["국명"].isin(selected_species)]

if selected_habitats:
    filtered = filtered[filtered["서식지명"].isin(selected_habitats)]

clustered_df = run_dbscan(filtered, eps_meters=eps_meters, min_samples=min_samples)

# =========================================================
# 대시보드
# =========================================================
if page == "대시보드":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("전체 출현 기록 수", f"{len(filtered):,}", "필터 적용 결과")
    with c2:
        metric_card("고유 종 수", f"{filtered['학명'].nunique():,}", "학명 기준")
    with c3:
        metric_card("고유 지역 수", f"{filtered['세부통계용명칭'].nunique():,}", "세부통계용명칭 기준")
    with c4:
        valid_clusters = clustered_df[clustered_df["cluster"] != -1]["cluster"].nunique()
        metric_card("검출 클러스터 수", f"{valid_clusters:,}", "DBSCAN 결과")

    st.markdown("")

    left_col, right_col = st.columns([0.9, 2.1])

    with left_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">현재 탐색 조건</div>', unsafe_allow_html=True)
        st.write(f"**연도 범위**: {year_range[0]} ~ {year_range[1]}" if year_range else "**연도 범위**: 없음")
        st.write(f"**선택 종 수**: {len(selected_species)}")
        st.write(f"**선택 서식지 수**: {len(selected_habitats)}")
        st.write(f"**원본 좌표계**: {source_crs}")
        st.write(f"**DBSCAN 설정**: eps {eps_meters}m / min_samples {min_samples}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">해석 가이드</div>', unsafe_allow_html=True)
        st.write("• Heatmap은 출현기록의 공간적 밀집 정도를 보여줍니다.")
        st.write("• 클러스터 중심은 DBSCAN으로 검출된 공간 군집의 대표 위치입니다.")
        st.write("• hotspot은 보전 우선순위의 후보일 수 있지만, 현장 검증과 조사 편향 검토가 필요합니다.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">서울 지도 기반 hotspot 시각화</div>', unsafe_allow_html=True)

        map_center = SEOUL_CENTER
        valid_pts = clustered_df.dropna(subset=["lat", "lon"])
        if not valid_pts.empty:
            map_center = [valid_pts["lat"].mean(), valid_pts["lon"].mean()]

        m = build_base_map(center=map_center, zoom_start=11)

        if show_points:
            add_occurrence_markers(m, clustered_df)

        if show_heatmap:
            add_heatmap(m, clustered_df)

        if show_cluster_centers:
            add_cluster_markers(m, clustered_df)

        add_custom_legend(m)
        folium.LayerControl(collapsed=True).add_to(m)
        st_folium(m, width=None, height=map_height)

        st.markdown('</div>', unsafe_allow_html=True)

    bottom_left, bottom_right = st.columns([1.15, 0.95])

    with bottom_left:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">통계 패널</div>', unsafe_allow_html=True)

        fig_bar = make_bar_species_by_region(filtered)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)

        fig_line = make_line_yearly(filtered)
        if fig_line:
            st.plotly_chart(fig_line, use_container_width=True)

        top_species = make_top_species_table(filtered)
        st.markdown("**상위 10개 출현 종**")
        if not top_species.empty:
            st.dataframe(top_species, use_container_width=True, hide_index=True)
        else:
            st.info("표시할 종 정보가 없습니다.")

        st.markdown('</div>', unsafe_allow_html=True)

    with bottom_right:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">클러스터 상세 정보</div>', unsafe_allow_html=True)

        cluster_options = sorted(
            clustered_df.loc[clustered_df["cluster"] != -1, "cluster"].dropna().unique().tolist()
        )
        cluster_options = [int(c) for c in cluster_options]

        if cluster_options:
            selected_cluster = st.selectbox("클러스터 선택", options=cluster_options)
            detail = get_cluster_detail(clustered_df, selected_cluster)

            if detail:
                st.markdown('<div class="soft-card">', unsafe_allow_html=True)
                st.write(f"**클러스터 ID**: {detail['cluster_id']}")
                st.write(f"**출현 기록 수**: {detail['records']}")
                st.write(f"**고유 종 수**: {detail['species_n']}")
                st.write(f"**출현년도 범위**: {detail['year_min']} ~ {detail['year_max']}")
                st.write(f"**대표 서식지**: {detail['대표서식지']}")
                st.write(f"**대표 지역**: {detail['대표지역']}")
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("**주요 종 구성**")
                st.dataframe(detail["species_table"], use_container_width=True, hide_index=True)
        else:
            st.info("현재 필터 조건에서는 검출된 클러스터가 없습니다. eps 또는 최소 샘플 수를 조정해보세요.")

        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 분석 페이지
# =========================================================
elif page == "분석":
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">클러스터 및 다양성 분석</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)

    with a1:
        cluster_size_df = (
            clustered_df[clustered_df["cluster"] != -1]
            .groupby("cluster")
            .size()
            .reset_index(name="출현기록수")
            .sort_values("출현기록수", ascending=False)
        )
        if not cluster_size_df.empty:
            fig = px.bar(
                cluster_size_df,
                x="cluster",
                y="출현기록수",
                title="클러스터별 출현 기록 수",
                text_auto=True,
            )
            fig.update_layout(
                xaxis_title="클러스터 ID",
                yaxis_title="출현 기록 수",
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("클러스터 분석 결과가 없습니다.")

    with a2:
        richness_df = (
            filtered.dropna(subset=["세부통계용명칭", "학명"])
            .groupby("세부통계용명칭")["학명"]
            .nunique()
            .reset_index(name="종풍부도")
            .sort_values("종풍부도", ascending=False)
            .head(20)
        )
        if not richness_df.empty:
            fig = px.bar(
                richness_df,
                x="세부통계용명칭",
                y="종풍부도",
                title="지역별 종 풍부도",
                text_auto=True,
            )
            fig.update_layout(
                xaxis_title="지역",
                yaxis_title="종 풍부도",
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("종 풍부도 분석 결과가 없습니다.")

    yearly_species_df = (
        filtered.dropna(subset=["출현년도", "학명"])
        .groupby("출현년도")["학명"]
        .nunique()
        .reset_index(name="고유종수")
        .sort_values("출현년도")
    )
    if not yearly_species_df.empty:
        fig = px.line(
            yearly_species_df,
            x="출현년도",
            y="고유종수",
            markers=True,
            title="연도별 고유 종 수",
        )
        fig.update_layout(
            xaxis_title="출현년도",
            yaxis_title="고유 종 수",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 데이터 정보 페이지
# =========================================================
elif page == "데이터 정보":
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">데이터 설명 및 유의사항</div>', unsafe_allow_html=True)

    info_df = pd.DataFrame({
        "컬럼명": ["종코드", "국명", "학명", "서식지코드", "서식지명", "세부통계용명칭", "출현년도", "원전", "X좌표", "Y좌표", "서식지비고정보"],
        "설명": [
            "종 식별 코드",
            "국문 종명",
            "학명",
            "서식지 식별 코드",
            "서식지명",
            "세부 지역명 또는 행정 단위",
            "출현 연도",
            "문헌 또는 출처",
            "원본 X 좌표",
            "원본 Y 좌표",
            "좌표계 및 비고 정보",
        ]
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)

    st.markdown("**해석 시 유의점**")
    st.write("• 문헌 기반 출현자료이므로 조사 강도 차이가 반영될 수 있습니다.")
    st.write("• absence data가 없기 때문에 미기록을 실제 부재로 해석하면 안 됩니다.")
    st.write("• hotspot은 관측 집중, 접근성, 조사 시기 차이의 결과일 수도 있습니다.")
    st.write("• 좌표계 설정이 맞지 않으면 지도상 위치가 크게 어긋날 수 있습니다.")

    st.markdown("**원본 데이터 미리보기**")
    st.dataframe(df.head(30), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 소개 페이지
# =========================================================
elif page == "소개":
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">플랫폼 소개</div>', unsafe_allow_html=True)
    st.write(
        "Seoul Biodiversity Explorer는 서울시 야생동식물 출현 데이터를 기반으로 "
        "생물다양성의 공간 분포와 hotspot 패턴을 직관적으로 탐색할 수 있도록 설계된 웹 플랫폼입니다."
    )
    st.write("주요 기능")
    st.write("• 지도 기반 출현 위치 시각화")
    st.write("• Heatmap 기반 hotspot 탐색")
    st.write("• DBSCAN 기반 공간 클러스터링")
    st.write("• 종 다양성 및 시계열 통계 분석")
    st.write("확장 방향")
    st.write("• 보호종/외래종 분리 분석")
    st.write("• 토지이용도 및 녹지연결성 자료 결합")
    st.write("• 자치구 단위 정책 지원형 대시보드 확장")
    st.markdown('</div>', unsafe_allow_html=True)
