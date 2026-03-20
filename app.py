import math
from typing import Optional

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
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="서울 생물다양성 Hotspot 분석 플랫폼",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 사용자 설정
# GitHub raw CSV URL로 교체해서 사용
# 예:
# https://raw.githubusercontent.com/USER/REPO/BRANCH/path/to/file.csv
# =========================================================
GITHUB_RAW_CSV_URL = "https://raw.githubusercontent.com/JinsKim-EcoEvo/Seoul_wildlife/f3f7e06d025e776da585e9960ecb38906ffe3e86/Seoul_wildlife.csv"

DEFAULT_SOURCE_CRS = "EPSG:2097"  # 동경측지계로 추정되는 기본값
SEOUL_CENTER = [37.5665, 126.9780]

# =========================================================
# 스타일
# =========================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        color: #1f4d3a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #4b6358;
        margin-bottom: 1rem;
    }
    .panel-card {
        background-color: #f7fbf8;
        border: 1px solid #dfece5;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .small-note {
        color: #5c7268;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 유틸
# =========================================================
def safe_read_csv(url: str) -> pd.DataFrame:
    """
    cp949 / utf-8-sig / euc-kr 순으로 시도
    """
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
    """
    좌표계 선택이 다소 틀렸을 때 엉뚱한 점이 섞이는 걸 완화
    """
    cond = (
        df["lat"].between(37.35, 37.75, inclusive="both")
        & df["lon"].between(126.70, 127.25, inclusive="both")
    )
    return df.loc[cond].copy()


def run_dbscan(df: pd.DataFrame, eps_meters: int, min_samples: int) -> pd.DataFrame:
    """
    위경도를 meter 기반 근사 clustering 하기 위해
    Web Mercator(EPSG:3857)로 변환 후 DBSCAN 수행
    """
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
    return f"""
    <div style="font-size:13px; line-height:1.5;">
        <b>국명</b>: {row.get('국명', '-') if pd.notna(row.get('국명', np.nan)) else '-'}<br>
        <b>학명</b>: {row.get('학명', '-') if pd.notna(row.get('학명', np.nan)) else '-'}<br>
        <b>출현년도</b>: {row.get('출현년도', '-') if pd.notna(row.get('출현년도', np.nan)) else '-'}<br>
        <b>서식지명</b>: {row.get('서식지명', '-') if pd.notna(row.get('서식지명', np.nan)) else '-'}<br>
        <b>세부지역</b>: {row.get('세부통계용명칭', '-') if pd.notna(row.get('세부통계용명칭', np.nan)) else '-'}<br>
        <b>원전</b>: {row.get('원전', '-') if pd.notna(row.get('원전', np.nan)) else '-'}
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


def add_occurrence_markers(m: folium.Map, df: pd.DataFrame, sample_n: int = 1500):
    points = df.dropna(subset=["lat", "lon"]).copy()
    if points.empty:
        return

    if len(points) > sample_n:
        points = points.sample(sample_n, random_state=42)

    cluster = MarkerCluster(name="종 출현 지점").add_to(m)

    for _, row in points.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            weight=1,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(make_popup_html(row), max_width=320),
            tooltip=row.get("국명", "출현 지점"),
        ).add_to(cluster)


def add_heatmap(m: folium.Map, df: pd.DataFrame):
    points = df.dropna(subset=["lat", "lon"])
    if points.empty:
        return
    heat_data = points[["lat", "lon"]].values.tolist()
    HeatMap(
        heat_data,
        name="Hotspot Heatmap",
        radius=18,
        blur=14,
        min_opacity=0.35,
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
        <div style="font-size:13px; line-height:1.5;">
            <b>클러스터 ID</b>: {int(row['cluster'])}<br>
            <b>출현 기록 수</b>: {int(row['count'])}<br>
            <b>고유 학명 수</b>: {int(row['species_n'])}
        </div>
        """
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup, max_width=260),
            tooltip=f"Cluster {int(row['cluster'])}",
            icon=folium.Icon(icon="info-sign"),
        ).add_to(layer)


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

    text_cols = ["종코드", "국명", "학명", "서식지코드", "서식지명", "세부통계용명칭", "원전", "서식지비고정보"]
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
    fig = px.bar(chart_df, x="세부통계용명칭", y="종수", title="상위 지역별 종수")
    fig.update_layout(xaxis_title="지역", yaxis_title="고유 종수")
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
    fig = px.line(chart_df, x="출현년도", y="출현기록수", markers=True, title="연도별 출현 기록 수")
    fig.update_layout(xaxis_title="출현년도", yaxis_title="출현 기록 수")
    return fig


def make_top_species_table(df: pd.DataFrame) -> pd.DataFrame:
    table_df = (
        df.dropna(subset=["학명"])
        .groupby(["국명", "학명"])
        .size()
        .reset_index(name="출현기록수")
        .sort_values("출현기록수", ascending=False)
        .head(10)
    )
    return table_df


def get_cluster_detail(df: pd.DataFrame, cluster_id: Optional[int]):
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

    detail = {
        "cluster_id": cluster_id,
        "records": len(cluster_df),
        "species_n": cluster_df["학명"].nunique(),
        "year_min": cluster_df["출현년도"].dropna().min() if cluster_df["출현년도"].notna().any() else "-",
        "year_max": cluster_df["출현년도"].dropna().max() if cluster_df["출현년도"].notna().any() else "-",
        "대표서식지": habitat_mode,
        "대표지역": region_mode,
        "species_table": species_table,
    }
    return detail


# =========================================================
# 상단 제목
# =========================================================
st.markdown('<div class="main-title">서울 생물다양성 Hotspot 분석 플랫폼</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">문헌 기반 야생동식물 출현 데이터를 활용해 공간적 집중지역과 생물다양성 패턴을 탐색합니다.</div>',
    unsafe_allow_html=True,
)

# =========================================================
# 상단 네비게이션
# =========================================================
page = st.radio(
    "페이지 선택",
    ["대시보드", "분석", "데이터 정보", "About"],
    horizontal=True,
    label_visibility="collapsed",
)

# =========================================================
# 사이드바
# =========================================================
with st.sidebar:
    st.header("필터 패널")

    st.markdown("#### 데이터 경로")
    github_url = st.text_input("GitHub Raw CSV URL", value=GITHUB_RAW_CSV_URL)

    st.markdown("#### 좌표계 설정")
    source_crs = st.selectbox(
        "원본 좌표계",
        options=["EPSG:2097", "EPSG:5174", "EPSG:5181", "EPSG:4326"],
        index=0,
        help="기본값은 동경측지계 추정치(EPSG:2097)입니다. 점 위치가 어긋나면 다른 좌표계를 시험해보세요.",
    )

# =========================================================
# 데이터 로드
# =========================================================
try:
    df = load_and_prepare_data(github_url, source_crs)
except Exception as e:
    st.error("CSV를 불러오지 못했습니다. GitHub raw URL과 파일 인코딩을 확인해줘.")
    st.exception(e)
    st.stop()

if df.empty:
    st.warning("불러온 데이터가 비어 있습니다.")
    st.stop()

# =========================================================
# 공통 필터
# =========================================================
all_years = sorted([int(y) for y in df["출현년도"].dropna().unique().tolist()])
all_species = sorted(df["국명"].dropna().unique().tolist())
all_habitats = sorted(df["서식지명"].dropna().unique().tolist())

with st.sidebar:
    st.markdown("#### 분석 필터")

    if all_years:
        year_range = st.slider(
            "출현년도 범위",
            min_value=min(all_years),
            max_value=max(all_years),
            value=(min(all_years), max(all_years)),
        )
    else:
        year_range = None
        st.info("출현년도 값이 없어 연도 필터를 비활성화합니다.")

    selected_species = st.multiselect(
        "종 선택 (국명 기준)",
        options=all_species,
        default=[],
    )

    selected_habitats = st.multiselect(
        "서식지 유형 선택",
        options=all_habitats,
        default=[],
    )

    show_points = st.checkbox("출현 지점 표시", value=True)
    show_heatmap = st.checkbox("Hotspot Heatmap 표시", value=True)
    show_cluster_centers = st.checkbox("클러스터 중심 표시", value=True)

    st.markdown("#### 클러스터링 설정")
    eps_meters = st.slider(
        "DBSCAN 거리 기준 eps (m)",
        min_value=100,
        max_value=3000,
        value=700,
        step=100,
    )
    min_samples = st.slider(
        "최소 샘플 수 min_samples",
        min_value=3,
        max_value=50,
        value=10,
        step=1,
    )

    st.markdown("#### 표시 옵션")
    map_height = st.slider("지도 높이", min_value=500, max_value=1000, value=680, step=20)

# 필터 적용
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
# 페이지 1: 대시보드
# =========================================================
if page == "대시보드":
    # ---- KPI
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("전체 출현 기록 수", f"{len(filtered):,}")
    k2.metric("고유 종 수", f"{filtered['학명'].nunique():,}")
    k3.metric("고유 지역 수", f"{filtered['세부통계용명칭'].nunique():,}")
    valid_clusters = clustered_df[clustered_df["cluster"] != -1]["cluster"].nunique()
    k4.metric("검출 클러스터 수", f"{valid_clusters:,}")

    st.markdown("---")

    # ---- 상단: 좌측 필터 요약 / 우측 지도
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("현재 필터 요약")
        st.write(f"**연도 범위**: {year_range[0]} ~ {year_range[1]}" if year_range else "연도 필터 없음")
        st.write(f"**선택 종 수**: {len(selected_species)}")
        st.write(f"**선택 서식지 수**: {len(selected_habitats)}")
        st.write(f"**좌표계**: {source_crs}")
        st.write(f"**eps / min_samples**: {eps_meters}m / {min_samples}")
        st.markdown(
            '<div class="small-note">좌표 위치가 서울 밖으로 보이면 좌표계 설정을 바꿔보는 것이 좋습니다.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("해석 포인트")
        st.write("- Heatmap은 출현기록 밀집도를 직관적으로 보여준다.")
        st.write("- DBSCAN 클러스터는 공간적으로 가까운 출현기록 군집을 도출한다.")
        st.write("- 클러스터가 곧 보전 우선순위를 의미하는 것은 아니므로 추가 해석이 필요하다.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.subheader("서울 지도 기반 Hotspot 시각화")

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

        folium.LayerControl(collapsed=False).add_to(m)
        map_data = st_folium(m, width=None, height=map_height, returned_objects=["last_object_clicked_tooltip"])

    st.markdown("---")

    # ---- 하단: 통계 패널 / 클러스터 상세 정보
    stat_col, detail_col = st.columns([1.1, 0.9])

    with stat_col:
        st.subheader("통계 패널")

        fig_bar = make_bar_species_by_region(filtered)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)

        fig_line = make_line_yearly(filtered)
        if fig_line:
            st.plotly_chart(fig_line, use_container_width=True)

        top_species = make_top_species_table(filtered)
        st.markdown("#### 상위 10개 출현 종")
        if not top_species.empty:
            st.dataframe(top_species, use_container_width=True, hide_index=True)
        else:
            st.info("표시할 종 정보가 없습니다.")

    with detail_col:
        st.subheader("클러스터 상세 정보")

        cluster_options = sorted(
            clustered_df.loc[clustered_df["cluster"] != -1, "cluster"].dropna().unique().tolist()
        )
        cluster_options = [int(c) for c in cluster_options]

        selected_cluster = None
        if cluster_options:
            selected_cluster = st.selectbox("클러스터 선택", options=cluster_options)
            detail = get_cluster_detail(clustered_df, selected_cluster)

            if detail:
                st.markdown('<div class="panel-card">', unsafe_allow_html=True)
                st.write(f"**클러스터 ID**: {detail['cluster_id']}")
                st.write(f"**출현 기록 수**: {detail['records']}")
                st.write(f"**고유 종 수**: {detail['species_n']}")
                st.write(f"**출현년도 범위**: {detail['year_min']} ~ {detail['year_max']}")
                st.write(f"**대표 서식지**: {detail['대표서식지']}")
                st.write(f"**대표 지역**: {detail['대표지역']}")
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("#### 주요 종 구성")
                st.dataframe(detail["species_table"], use_container_width=True, hide_index=True)
        else:
            st.info("현재 조건에서 검출된 클러스터가 없습니다. eps 또는 min_samples를 조정해봐.")

# =========================================================
# 페이지 2: 분석
# =========================================================
elif page == "분석":
    st.subheader("클러스터 분석 / 종 다양성 분석 / 시계열 분석")

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
            fig = px.bar(cluster_size_df, x="cluster", y="출현기록수", title="클러스터별 출현 기록 수")
            fig.update_layout(xaxis_title="클러스터 ID", yaxis_title="출현 기록 수")
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
            fig = px.bar(richness_df, x="세부통계용명칭", y="종풍부도", title="지역별 종 풍부도")
            fig.update_layout(xaxis_title="지역", yaxis_title="종 풍부도")
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
        fig = px.line(yearly_species_df, x="출현년도", y="고유종수", markers=True, title="연도별 고유 종 수")
        fig.update_layout(xaxis_title="출현년도", yaxis_title="고유 종 수")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 페이지 3: 데이터 정보
# =========================================================
elif page == "데이터 정보":
    st.subheader("데이터 설명 및 주의사항")

    st.markdown("### 컬럼 구성")
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
            "좌표계 등 비고 정보",
        ]
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)

    st.markdown("### 데이터 해석 시 유의점")
    st.write("- 이 데이터는 문헌 기반 출현자료이므로 sampling bias가 존재할 수 있다.")
    st.write("- absence data가 없으므로 미출현을 실제 부재로 해석하면 안 된다.")
    st.write("- hotspot은 관측 집중 또는 조사 집중의 결과일 수 있어 현장 검증이 필요하다.")
    st.write("- 좌표계가 다르면 지도 위치가 달라질 수 있으므로 원본 메타데이터 확인이 중요하다.")

    st.markdown("### 원본 데이터 미리보기")
    st.dataframe(df.head(30), use_container_width=True)

# =========================================================
# 페이지 4: About
# =========================================================
elif page == "About":
    st.subheader("플랫폼 소개")
    st.write(
        """
        이 플랫폼은 서울시 야생동식물 출현자료를 기반으로
        생물다양성의 공간적 패턴과 hotspot을 탐색하기 위한 웹 기반 분석 도구입니다.
        """
    )
    st.write(
        """
        주요 기능:
        - 지도 기반 출현 기록 시각화
        - Heatmap 기반 hotspot 탐색
        - DBSCAN 기반 공간 클러스터링
        - 종 다양성 및 시계열 통계 분석
        """
    )
    st.write(
        """
        연구 확장 방향:
        - 보호종/외래종 분리 분석
        - 토지이용도 및 녹지연결성 자료 결합
        - 자치구 또는 서식지 유형별 비교 분석
        """
    )
