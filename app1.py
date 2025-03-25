import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# 페이지 설정
st.set_page_config(page_title="제품 품질관리 대시보드", layout="wide")

# 세션 초기화 기능
def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.sidebar.success('세션이 초기화되었습니다. 페이지를 새로고침하세요.')
    st.stop()

if st.sidebar.button('세션 초기화'):
    reset_session()

# 사이드바 설정
st.sidebar.title("품질관리 시스템")
app_mode = st.sidebar.selectbox("메뉴 선택", ["대시보드", "데이터 업로드", "품질 분석", "불량 추적", "설정"], index=0)


# 샘플 데이터 생성 함수
def generate_sample_data():
    np.random.seed(42)
    dates = [(datetime.now() - timedelta(days=x)).strftime("%Y-%m-%d") for x in range(100)]

    data = {
        "date": dates,
        "product_id": np.random.choice(["A001", "A002", "A003", "B001", "B002", "C001"], 100),
        "production_line": np.random.choice(["라인1", "라인2", "라인3"], 100),
        "inspector": np.random.choice(["김검사", "이검사", "박검사", "최검사"], 100),
        "weight_g": np.random.normal(100, 10, 100).round(2),
        "length_mm": np.random.normal(150, 5, 100).round(2),
        "width_mm": np.random.normal(75, 7, 100).round(2),
        "height_mm": np.random.normal(50, 5, 100).round(2),
        "is_defective": np.random.choice([0, 1], 100, p=[0.60, 0.40]),
        "defect_type": [""] * 100,
    }

    # 불량인 경우에만 불량유형 지정
    defect_types = ["치수불량", "외관불량", "기능불량", "포장불량"]
    for i in range(100):
        if data["is_defective"][i] == 1:
            data["defect_type"][i] = np.random.choice(defect_types)

    return pd.DataFrame(data)


# 세션 초기화 및 데이터 확인
if 'data' not in st.session_state:
    st.session_state.data = generate_sample_data()
else:
    # 기존 세션 데이터에 필요한 컬럼이 없는 경우 새로 생성
    required_columns = ["date", "product_id", "production_line", "inspector",
                        "weight_g", "length_mm", "width_mm", "height_mm",
                        "is_defective", "defect_type"]

    # 필요한 컬럼이 없으면 데이터 재생성
    missing_columns = [col for col in required_columns if col not in st.session_state.data.columns]
    if missing_columns:
        st.sidebar.warning(f"세션에 필요한 컬럼이 누락되어 있어 데이터를 재생성합니다: {', '.join(missing_columns)}")
        st.session_state.data = generate_sample_data()


# 제목 및 설명
def display_header():
    st.title("제품 품질관리 시스템")
    st.markdown("이 앱은 제품 생산 과정의 품질을 모니터링하고 분석하는 데 사용됩니다.")


# 안전하게 데이터프레임 컬럼에 접근하기 위한 함수
def safe_get_column(df, column_name, default_value=None):
    if column_name in df.columns:
        return df[column_name]
    else:
        if default_value is None:
            return pd.Series([0] * len(df))
        return pd.Series([default_value] * len(df))


# 대시보드 페이지
def dashboard_page():
    display_header()

    # 컬럼 확인
    if "date" not in st.session_state.data.columns:
        st.error("필요한 컬럼이 누락되었습니다. '데이터 업로드' 메뉴에서 데이터를 다시 생성해주세요.")
        if st.button("샘플 데이터 다시 생성"):
            st.session_state.data = generate_sample_data()
            st.experimental_rerun()
        return

    # 메트릭 카드 표시
    col1, col2, col3, col4 = st.columns(4)

    recent_data = st.session_state.data.sort_values("date", ascending=False)
    total_products = len(recent_data)
    defect_count = sum(safe_get_column(recent_data, "is_defective"))
    defect_rate = (defect_count / total_products * 100) if total_products > 0 else 0

    col1.metric("총 검사 제품", f"{total_products}개")
    col2.metric("불량 제품", f"{defect_count}개")
    col3.metric("불량률", f"{defect_rate:.2f}%")

    latest_date = recent_data["date"].iloc[0] if not recent_data.empty else "데이터 없음"
    col4.metric("최근 데이터", latest_date)

    # 불량률 추세 차트
    st.subheader("일별 불량률 추세")
    daily_defect = recent_data.groupby("date")["is_defective"].mean() * 100
    daily_defect = daily_defect.reset_index()
    daily_defect.columns = ["날짜", "불량률(%)"]

    fig = px.line(daily_defect, x="날짜", y="불량률(%)", markers=True)
    # 목표 불량률 추가 (5%)
    fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="목표 불량률 5%")
    st.plotly_chart(fig, use_container_width=True)

    # 불량 유형 분석
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("불량 유형 분포")
        defect_data = recent_data[recent_data["is_defective"] == 1]
        if not defect_data.empty:
            defect_type_count = defect_data["defect_type"].value_counts().reset_index()
            defect_type_count.columns = ["불량유형", "개수"]

            fig = px.pie(defect_type_count, values="개수", names="불량유형", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("불량 데이터가 없습니다.")

    with col2:
        st.subheader("생산라인별 불량률")
        line_defect = recent_data.groupby("production_line")["is_defective"].mean() * 100
        line_defect = line_defect.reset_index()
        line_defect.columns = ["생산라인", "불량률(%)"]

        fig = px.bar(line_defect, x="생산라인", y="불량률(%)", color="불량률(%)",
                     color_continuous_scale=["green", "yellow", "red"])
        st.plotly_chart(fig, use_container_width=True)

    # 검사 항목별 분포
    st.subheader("품질 측정값 분포")

    # 측정 항목 선택
    numeric_cols = [col for col in ["weight_g", "length_mm", "width_mm", "height_mm"]
                    if col in recent_data.columns]

    if numeric_cols:
        measurement = st.selectbox("측정 항목 선택", numeric_cols)

        # 히스토그램과 박스 플롯
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(recent_data, x=measurement, color="is_defective",
                               barmode="overlay", nbins=20,
                               color_discrete_map={0: "green", 1: "red"})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(recent_data, x="production_line", y=measurement, color="production_line")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("측정값 데이터가 존재하지 않습니다.")


# 데이터 업로드 페이지
def data_upload_page():
    st.header("데이터 업로드")

    upload_option = st.radio("데이터 소스 선택", ["샘플 데이터 사용", "CSV 파일 업로드"])

    if upload_option == "샘플 데이터 사용":
        if st.button("샘플 데이터 생성"):
            st.session_state.data = generate_sample_data()
            st.success("샘플 데이터가 생성되었습니다!")
            st.dataframe(st.session_state.data)
    else:
        uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)

                # 필수 컬럼 확인 및 추가
                required_columns = ["date", "product_id", "production_line", "inspector",
                                    "weight_g", "length_mm", "width_mm", "height_mm",
                                    "is_defective", "defect_type"]

                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    st.warning(f"다음 컬럼이 CSV 파일에 없어 자동으로 추가됩니다: {', '.join(missing_columns)}")

                    for col in missing_columns:
                        if col == "date":
                            data[col] = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                                         for i in range(len(data))]
                        elif col == "is_defective":
                            data[col] = 0
                        else:
                            data[col] = ""

                st.session_state.data = data
                st.success("파일 업로드 성공!")
                st.dataframe(data)
            except Exception as e:
                st.error(f"파일 업로드 중 오류 발생: {e}")

    # 데이터 다운로드 옵션
    if 'data' in st.session_state and not st.session_state.data.empty:
        st.subheader("데이터 다운로드")
        csv = st.session_state.data.to_csv(index=False)
        st.download_button(
            label="CSV로 다운로드",
            data=csv,
            file_name="quality_control_data.csv",
            mime="text/csv",
        )


# X-bar R 차트 생성 함수
def create_xbar_r_chart(data, metric, subgroup_size=5):
    """
    X-bar R 차트를 생성하는 함수

    Parameters:
    data (DataFrame): 데이터
    metric (str): 측정 항목 (컬럼명)
    subgroup_size (int): 부분군 크기

    Returns:
    fig_xbar, fig_r: X-bar 차트와 R 차트 Figure 객체
    """
    # 데이터 정렬 (날짜순)
    sorted_data = data.sort_values("date").reset_index(drop=True)

    # 부분군 생성
    n_subgroups = len(sorted_data) // subgroup_size

    if n_subgroups == 0:
        return None, None, None

    subgroups = []
    subgroup_names = []

    for i in range(n_subgroups):
        start_idx = i * subgroup_size
        end_idx = start_idx + subgroup_size

        # 부분군 데이터
        subgroup = sorted_data.iloc[start_idx:end_idx][metric].values
        subgroups.append(subgroup)

        # 부분군 이름 (첫 날짜 ~ 마지막 날짜)
        first_date = sorted_data.iloc[start_idx]["date"]
        last_date = sorted_data.iloc[min(end_idx - 1, len(sorted_data) - 1)]["date"]
        subgroup_names.append(f"{first_date} ~ {last_date}")


    # 계산
    means = [np.mean(subgroup) for subgroup in subgroups]
    ranges = [np.max(subgroup) - np.min(subgroup) for subgroup in subgroups]

    # 전체 평균 및 범위
    mean_of_means = np.mean(means)
    mean_of_ranges = np.mean(ranges)

    # 관리 한계선 계산
    # X-bar 차트 관리 한계
    a2_values = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419}
    a2 = a2_values.get(subgroup_size, 0.577)  # 기본값 5

    x_ucl = mean_of_means + a2 * mean_of_ranges
    x_lcl = mean_of_means - a2 * mean_of_ranges

    # R 차트 관리 한계
    d3_values = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076}
    d4_values = {2: 3.267, 3: 2.575, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924}

    d3 = d3_values.get(subgroup_size, 0)  # 기본값 5
    d4 = d4_values.get(subgroup_size, 2.114)  # 기본값 5

    r_ucl = d4 * mean_of_ranges
    r_lcl = d3 * mean_of_ranges

    # X-bar 차트 생성
    fig_xbar = go.Figure()

    # X-bar 데이터 포인트
    fig_xbar.add_trace(go.Scatter(
        x=list(range(1, n_subgroups + 1)),
        y=means,
        mode='lines+markers',
        name='X-bar'
    ))

    # 중심선 (전체 평균)
    fig_xbar.add_trace(go.Scatter(
        x=list(range(1, n_subgroups + 1)),
        y=[mean_of_means] * n_subgroups,
        mode='lines',
        name='중심선 (X-bar)',
        line=dict(color='green', dash='dash')
    ))

    # 상한/하한 관리 한계
    fig_xbar.add_trace(go.Scatter(
        x=list(range(1, n_subgroups + 1)),
        y=[x_ucl] * n_subgroups,
        mode='lines',
        name='UCL (X-bar)',
        line=dict(color='red')
    ))

    fig_xbar.add_trace(go.Scatter(
        x=list(range(1, n_subgroups + 1)),
        y=[x_lcl] * n_subgroups,
        mode='lines',
        name='LCL (X-bar)',
        line=dict(color='red')
    ))

    fig_xbar.update_layout(
        title=f"{metric} X-bar 차트 (부분군 크기: {subgroup_size})",
        xaxis_title="부분군 번호",
        yaxis_title=f"{metric} 평균",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, n_subgroups + 1)),
            ticktext=subgroup_names
        ),
        xaxis_tickangle=-45
    )

    # R 차트 생성
    fig_r = go.Figure()

    # R 데이터 포인트
    fig_r.add_trace(go.Scatter(
        x=list(range(1, n_subgroups + 1)),
        y=ranges,
        mode='lines+markers',
        name='R'
    ))

    # 중심선 (전체 범위 평균)
    fig_r.add_trace(go.Scatter(
        x=list(range(1, n_subgroups + 1)),
        y=[mean_of_ranges] * n_subgroups,
        mode='lines',
        name='중심선 (R)',
        line=dict(color='green', dash='dash')
    ))

    # 상한/하한 관리 한계
    fig_r.add_trace(go.Scatter(
        x=list(range(1, n_subgroups + 1)),
        y=[r_ucl] * n_subgroups,
        mode='lines',
        name='UCL (R)',
        line=dict(color='red')
    ))

    if r_lcl > 0:  # 하한선이 0보다 큰 경우에만 표시
        fig_r.add_trace(go.Scatter(
            x=list(range(1, n_subgroups + 1)),
            y=[r_lcl] * n_subgroups,
            mode='lines',
            name='LCL (R)',
            line=dict(color='red')
        ))

    fig_r.update_layout(
        title=f"{metric} R 차트 (부분군 크기: {subgroup_size})",
        xaxis_title="부분군 번호",
        yaxis_title=f"{metric} 범위",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, n_subgroups + 1)),
            ticktext=subgroup_names
        ),
        xaxis_tickangle=-45
    )

    # 기본 통계량 반환
    stats = {
        "X-bar 중심선": mean_of_means,
        "X-bar UCL": x_ucl,
        "X-bar LCL": x_lcl,
        "R 중심선": mean_of_ranges,
        "R UCL": r_ucl,
        "R LCL": r_lcl,
    }

    return fig_xbar, fig_r, stats


# 품질 분석 페이지
def quality_analysis_page():
    st.header("품질 분석")

    if 'data' not in st.session_state or st.session_state.data.empty:
        st.warning("분석할 데이터가 없습니다. 데이터 업로드 페이지에서 데이터를 추가해주세요.")
        return

    # 필수 컬럼 확인
    required_columns = ["date", "product_id", "is_defective"]
    missing_columns = [col for col in required_columns if col not in st.session_state.data.columns]

    if missing_columns:
        st.error(f"분석에 필요한 컬럼이 누락되었습니다: {', '.join(missing_columns)}")
        if st.button("샘플 데이터 다시 생성"):
            st.session_state.data = generate_sample_data()
            st.experimental_rerun()
        return

    # 제품군 필터링
    product_filter = st.multiselect(
        "제품 ID 선택",
        options=st.session_state.data["product_id"].unique(),
        default=st.session_state.data["product_id"].unique()
    )

    filtered_data = st.session_state.data[st.session_state.data["product_id"].isin(product_filter)]

    # 시간 범위 필터링
    try:
        date_range = st.date_input(
            "날짜 범위 선택",
            value=(
                datetime.strptime(filtered_data["date"].min(), "%Y-%m-%d").date(),
                datetime.strptime(filtered_data["date"].max(), "%Y-%m-%d").date()
            ),
            key="date_range"
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (filtered_data["date"] >= start_date.strftime("%Y-%m-%d")) & (
                        filtered_data["date"] <= end_date.strftime("%Y-%m-%d"))
            filtered_data = filtered_data[mask]
    except:
        st.warning("날짜 데이터 형식에 문제가 있습니다. 전체 데이터를 표시합니다.")

    # 통계 분석
    st.subheader("기본 통계 분석")

    numeric_cols = [col for col in ["weight_g", "length_mm", "width_mm", "height_mm"]
                    if col in filtered_data.columns]

    if numeric_cols:
        stats_df = filtered_data[numeric_cols].describe().T
        stats_df = stats_df.round(2)
        st.dataframe(stats_df)

        # 탭으로 관리도와 X-bar R 차트 구분
        tabs = st.tabs(["관리도 (I-MR Chart)", "X-bar R 차트"])

        with tabs[0]:
            # 관리도 (I-MR Chart)
            st.subheader("관리도 (I-MR Chart)")

            metric = st.selectbox("측정 항목", numeric_cols, key="imr_metric")

            # 데이터 준비
            chart_data = filtered_data.sort_values("date")[["date", metric]].reset_index(drop=True)

            # 관리 한계선 계산
            mean_val = chart_data[metric].mean()
            std_val = chart_data[metric].std()
            ucl = mean_val + 3 * std_val  # 상한 관리 한계
            lcl = mean_val - 3 * std_val  # 하한 관리 한계

            # 관리도 그리기
            fig = go.Figure()

            # 데이터 포인트
            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data[metric],
                mode='lines+markers',
                name=metric
            ))

            # 중심선 (평균)
            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=[mean_val] * len(chart_data),
                mode='lines',
                name='평균',
                line=dict(color='green', dash='dash')
            ))

            # 상한/하한 관리 한계
            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=[ucl] * len(chart_data),
                mode='lines',
                name='UCL (상한 관리 한계)',
                line=dict(color='red')
            ))

            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=[lcl] * len(chart_data),
                mode='lines',
                name='LCL (하한 관리 한계)',
                line=dict(color='red')
            ))

            fig.update_layout(
                title=f"{metric} 관리도",
                xaxis_title="샘플 번호",
                yaxis_title=metric,
            )

            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            # X-bar R 차트
            st.subheader("X-bar R 차트")

            metric = st.selectbox("측정 항목", numeric_cols, key="xbar_metric")
            subgroup_size = st.slider("부분군 크기", min_value=2, max_value=10, value=5)

            # X-bar R 차트 생성
            fig_xbar, fig_r, stats = create_xbar_r_chart(filtered_data, metric, subgroup_size)

            if fig_xbar is None:
                st.warning(f"X-bar R 차트를 생성하기 위한 충분한 데이터가 없습니다. 부분군 크기({subgroup_size})보다 많은 데이터가 필요합니다.")
            else:
                # 통계량 표시
                st.write("### 관리 한계 통계량")
                stats_df = pd.DataFrame([stats])
                st.dataframe(stats_df.T)

                # X-bar 차트 표시
                st.plotly_chart(fig_xbar, use_container_width=True)

                # R 차트 표시
                st.plotly_chart(fig_r, use_container_width=True)

                # 해석 안내
                st.write("### X-bar R 차트 해석 가이드")
                st.markdown("""
                - **X-bar 차트**: 부분군 평균의 변화를 모니터링하여 공정의 중심 이동을 감지합니다.
                - **R 차트**: 부분군 내 범위(최대값-최소값)를 모니터링하여 공정의 변동성을 감지합니다.
                - **관리 상태**: 모든 포인트가 관리 한계선(UCL, LCL) 내에 있고 특별한 패턴이 없으면 공정이 관리 상태에 있다고 판단합니다.
                - **비정상 패턴**:
                  - 관리 한계선을 벗어난 포인트
                  - 중심선 한쪽에 연속 7개 이상의 포인트
                  - 연속적으로 증가하거나 감소하는 7개 이상의 포인트
                  - 주기적 패턴
                """)

        # 상관관계 분석
        if len(numeric_cols) > 1:
            st.subheader("측정 항목 간 상관관계")

            corr = filtered_data[numeric_cols].corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                aspect="auto",
                title="상관관계 히트맵"
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("분석할 수치 데이터가 없습니다.")


# 불량 추적 페이지
def defect_tracking_page():
    st.header("불량 추적")

    if 'data' not in st.session_state or st.session_state.data.empty:
        st.warning("추적할 데이터가 없습니다. 데이터 업로드 페이지에서 데이터를 추가해주세요.")
        return

    # 필수 컬럼 확인
    required_columns = ["date", "is_defective", "defect_type", "production_line"]
    missing_columns = [col for col in required_columns if col not in st.session_state.data.columns]

    if missing_columns:
        st.error(f"분석에 필요한 컬럼이 누락되었습니다: {', '.join(missing_columns)}")
        if st.button("샘플 데이터 다시 생성"):
            st.session_state.data = generate_sample_data()
            st.experimental_rerun()
        return

    # 불량 데이터만 필터링
    defect_data = st.session_state.data[st.session_state.data["is_defective"] == 1]

    if defect_data.empty:
        st.info("등록된 불량 제품이 없습니다.")
        return

    # 불량 요약 정보
    st.subheader("불량 요약")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("총 불량 수", len(defect_data))

        # 불량 유형별 카운트
        defect_types = defect_data["defect_type"].value_counts()
        st.write("불량 유형별 개수")
        st.dataframe(defect_types.reset_index().rename(columns={"defect_type": "불량유형", "count": "개수"}))

    with col2:
        st.write("생산라인별 불량 분포")
        line_defects = defect_data["production_line"].value_counts()
        fig = px.pie(
            values=line_defects.values,
            names=line_defects.index,
            title="생산라인별 불량 분포"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 시간대별 불량 추세
    st.subheader("시간대별 불량 추세")

    time_defects = defect_data.groupby("date").size().reset_index()
    time_defects.columns = ["날짜", "불량건수"]

    fig = px.line(
        time_defects,
        x="날짜",
        y="불량건수",
        markers=True,
        title="일별 불량 발생 추세"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 불량 유형별 시간 추세
    st.subheader("불량 유형별 시간 추세")

    if not defect_data.empty and "defect_type" in defect_data.columns and "date" in defect_data.columns:
        # 날짜와 불량 유형으로 그룹화
        type_time_defects = defect_data.groupby(["date", "defect_type"]).size().reset_index()
        type_time_defects.columns = ["날짜", "불량유형", "불량건수"]

        fig = px.line(
            type_time_defects,
            x="날짜",
            y="불량건수",
            color="불량유형",
            markers=True,
            title="불량 유형별 발생 추세"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 파레토 차트 (80/20 법칙)
    st.subheader("불량 유형 파레토 분석")

    if not defect_data.empty and "defect_type" in defect_data.columns:
        # 불량 유형별 개수 계산
        pareto_data = defect_data["defect_type"].value_counts().reset_index()
        pareto_data.columns = ["불량유형", "개수"]

        # 내림차순 정렬
        pareto_data = pareto_data.sort_values("개수", ascending=False)

        # 누적 백분율 계산
        pareto_data["누적개수"] = pareto_data["개수"].cumsum()
        pareto_data["누적비율"] = (pareto_data["누적개수"] / pareto_data["개수"].sum() * 100).round(1)

        # 파레토 차트 생성
        fig = go.Figure()

        # 막대 그래프 (불량 유형별 개수)
        fig.add_trace(go.Bar(
            x=pareto_data["불량유형"],
            y=pareto_data["개수"],
            name="불량건수"
        ))

        # 선 그래프 (누적 백분율)
        fig.add_trace(go.Scatter(
            x=pareto_data["불량유형"],
            y=pareto_data["누적비율"],
            mode="lines+markers",
            name="누적비율(%)",
            yaxis="y2"
        ))

        # 레이아웃 설정
        fig.update_layout(
            title="불량 유형 파레토 차트",
            xaxis=dict(title="불량유형"),
            yaxis=dict(title="불량건수"),
            yaxis2=dict(
                title="누적비율(%)",
                titlefont=dict(color="red"),
                tickfont=dict(color="red"),
                overlaying="y",
                side="right",
                range=[0, 100]
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**파레토 분석 결과**")
        st.dataframe(pareto_data)

    # 불량 세부 정보 테이블
    st.subheader("불량 세부 정보")

    # 테이블에 표시할 컬럼 선택
    display_cols = ["date", "product_id", "production_line", "inspector", "defect_type"]
    display_cols.extend([col for col in defect_data.columns if col.endswith("_mm") or col.endswith("_g")])

    # 존재하는 컬럼만 선택
    display_cols = [col for col in display_cols if col in defect_data.columns]

    if display_cols:
        st.dataframe(defect_data[display_cols])
    else:
        st.warning("표시할 데이터가 없습니다.")


# 설정 페이지
def settings_page():
    st.header("설정")

    st.subheader("품질 관리 기준")

    # 품질 기준 설정
    with st.form("quality_standards"):
        st.write("제품 측정 기준치 설정")

        col1, col2 = st.columns(2)

        with col1:
            weight_min = st.number_input("무게 최소값 (g)", value=95.0, step=0.1)
            weight_max = st.number_input("무게 최대값 (g)", value=105.0, step=0.1)

            length_min = st.number_input("길이 최소값 (mm)", value=145.0, step=0.1)
            length_max = st.number_input("길이 최대값 (mm)", value=155.0, step=0.1)

        with col2:
            width_min = st.number_input("너비 최소값 (mm)", value=72.0, step=0.1)
            width_max = st.number_input("너비 최대값 (mm)", value=78.0, step=0.1)

            height_min = st.number_input("높이 최소값 (mm)", value=48.0, step=0.1)
            height_max = st.number_input("높이 최대값 (mm)", value=52.0, step=0.1)

        st.write("불량률 목표 설정")
        target_defect_rate = st.slider("목표 불량률 (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

        if st.form_submit_button("설정 저장"):
            # 설정 값 저장
            if 'settings' not in st.session_state:
                st.session_state.settings = {}

            st.session_state.settings.update({
                "weight_min": weight_min,
                "weight_max": weight_max,
                "length_min": length_min,
                "length_max": length_max,
                "width_min": width_min,
                "width_max": width_max,
                "height_min": height_min,
                "height_max": height_max,
                "target_defect_rate": target_defect_rate
            })

            st.success("설정이 저장되었습니다!")

    # 사용자 정보 설정
    st.subheader("사용자 정보")

    with st.form("user_settings"):
        company_name = st.text_input("회사명", value="품질관리 주식회사")
        department = st.text_input("부서명", value="품질관리팀")
        admin_name = st.text_input("관리자명", value="김관리")

        if st.form_submit_button("사용자 정보 저장"):
            if 'user_info' not in st.session_state:
                st.session_state.user_info = {}

            st.session_state.user_info.update({
                "company_name": company_name,
                "department": department,
                "admin_name": admin_name
            })

            st.success("사용자 정보가 저장되었습니다!")

    # 앱 테마 설정
    st.subheader("앱 테마 설정")
    theme = st.selectbox("테마 선택", ["라이트", "다크"], index=0)

    if st.button("테마 적용"):
        # 실제로는 Streamlit에서 직접적인 테마 변경은 지원하지 않음
        st.info("테마 설정은 Streamlit의 설정 파일(.streamlit/config.toml)에서 변경할 수 있습니다.")


# 메인 앱 실행 로직
def main():
    if app_mode == "대시보드":
        dashboard_page()
    elif app_mode == "데이터 업로드":
        data_upload_page()
    elif app_mode == "품질 분석":
        quality_analysis_page()
    elif app_mode == "불량 추적":
        defect_tracking_page()
    elif app_mode == "설정":
        settings_page()


if __name__ == "__main__":
    main()