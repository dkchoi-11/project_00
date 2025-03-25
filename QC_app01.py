import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

# 페이지 설정
st.set_page_config(page_title="품질관리 대시보드", layout="wide")

# 헤더 및 설명
st.title("품질관리 대시보드")
st.markdown("관리번호별 측정값과 SPEC 기준을 기반으로 한 품질 분석 대시보드입니다.")

# 파일 업로드 위젯
uploaded_file = st.file_uploader("Excel 파일 업로드 (test_data.xlsx)", type=["xlsx"])


# 데이터 처리 함수
def load_data(file):
    """Excel 파일에서 데이터를 로드하는 함수"""
    # toLGE 및 SPEC 시트 읽기
    df_toLGE = pd.read_excel(file, sheet_name="toLGE")
    df_SPEC = pd.read_excel(file, sheet_name="SPEC")

    # 측정일자 열을 datetime 타입으로 변환
    df_toLGE['측정일자'] = pd.to_datetime(df_toLGE['측정일자'])

    return df_toLGE, df_SPEC


def create_control_chart(df, spec_df, management_number, chart_type='individual'):
    """관리도 차트를 생성하는 함수"""
    # 선택된 관리번호에 해당하는 데이터 필터링
    filtered_df = df[df['관리번호'] == management_number].copy()
    filtered_df = filtered_df.sort_values('측정일자')

    # SPEC 정보 가져오기
    spec_info = spec_df[spec_df['관리번호'] == management_number].iloc[0]
    usl = spec_info['USL']
    lsl = spec_info['LSL']
    target = spec_info['Target']
    ucl = spec_info['UCL'] if 'UCL' in spec_info else None
    lcl = spec_info['LCL'] if 'LCL' in spec_info else None

    # 관리항목명 가져오기
    item_name = filtered_df['CTQ/P 관리항목명'].iloc[0]

    # 플롯 생성
    fig = make_subplots(rows=1, cols=1)

    # 측정값 추가
    fig.add_trace(
        go.Scatter(
            x=filtered_df['측정일자'],
            y=filtered_df['측정값'],
            mode='lines+markers',
            name='측정값',
            line=dict(color='blue')
        )
    )

    # 중심선 (Target) 추가
    fig.add_trace(
        go.Scatter(
            x=[filtered_df['측정일자'].min(), filtered_df['측정일자'].max()],
            y=[target, target],
            mode='lines',
            name='Target',
            line=dict(color='green', dash='dash')
        )
    )

    # UCL, LCL 추가 (있는 경우)
    if ucl is not None:
        fig.add_trace(
            go.Scatter(
                x=[filtered_df['측정일자'].min(), filtered_df['측정일자'].max()],
                y=[ucl, ucl],
                mode='lines',
                name='UCL',
                line=dict(color='red', dash='dash')
            )
        )

    if lcl is not None:
        fig.add_trace(
            go.Scatter(
                x=[filtered_df['측정일자'].min(), filtered_df['측정일자'].max()],
                y=[lcl, lcl],
                mode='lines',
                name='LCL',
                line=dict(color='red', dash='dash')
            )
        )

    # USL, LSL 추가
    fig.add_trace(
        go.Scatter(
            x=[filtered_df['측정일자'].min(), filtered_df['측정일자'].max()],
            y=[usl, usl],
            mode='lines',
            name='USL',
            line=dict(color='purple')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[filtered_df['측정일자'].min(), filtered_df['측정일자'].max()],
            y=[lsl, lsl],
            mode='lines',
            name='LSL',
            line=dict(color='purple')
        )
    )

    # 기준 초과 데이터 하이라이트
    out_of_spec = filtered_df[(filtered_df['측정값'] > usl) | (filtered_df['측정값'] < lsl)]
    if not out_of_spec.empty:
        fig.add_trace(
            go.Scatter(
                x=out_of_spec['측정일자'],
                y=out_of_spec['측정값'],
                mode='markers',
                name='SPEC 초과',
                marker=dict(color='red', size=10, symbol='x')
            )
        )

    # 그래프 레이아웃 설정
    fig.update_layout(
        title=f"{item_name} 관리도 ({management_number})",
        xaxis_title="측정일자",
        yaxis_title="측정값",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def calculate_process_capability(df, spec_df, management_number):
    """공정능력지수(Cp, Cpk)를 계산하는 함수"""
    # 선택된 관리번호에 해당하는 데이터 필터링
    filtered_df = df[df['관리번호'] == management_number]

    # SPEC 정보 가져오기
    spec_info = spec_df[spec_df['관리번호'] == management_number].iloc[0]
    usl = spec_info['USL']
    lsl = spec_info['LSL']
    target = spec_info['Target']

    # 측정값 통계
    values = filtered_df['측정값'].values
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # 표본 표준편차

    # 공정능력지수 계산
    cp = (usl - lsl) / (6 * std) if std > 0 else np.nan
    cpu = (usl - mean) / (3 * std) if std > 0 else np.nan
    cpl = (mean - lsl) / (3 * std) if std > 0 else np.nan
    cpk = min(cpu, cpl) if (not np.isnan(cpu) and not np.isnan(cpl)) else np.nan

    # 공정 성능지수 계산 (표본 전체 기준)
    pp = (usl - lsl) / (6 * std) if std > 0 else np.nan
    ppu = (usl - mean) / (3 * std) if std > 0 else np.nan
    ppl = (mean - lsl) / (3 * std) if std > 0 else np.nan
    ppk = min(ppu, ppl) if (not np.isnan(ppu) and not np.isnan(ppl)) else np.nan

    # 공정 중심도
    k = abs(mean - target) / ((usl - lsl) / 2) if (usl - lsl) > 0 else np.nan

    return {
        'Mean': mean,
        'StdDev': std,
        'USL': usl,
        'LSL': lsl,
        'Target': target,
        'Cp': cp,
        'Cpu': cpu,
        'Cpl': cpl,
        'Cpk': cpk,
        'Pp': pp,
        'Ppu': ppu,
        'Ppl': ppl,
        'Ppk': ppk,
        'K': k
    }


def create_histogram(df, spec_df, management_number):
    """히스토그램과 정규분포 커브를 생성하는 함수"""
    # 선택된 관리번호에 해당하는 데이터 필터링
    filtered_df = df[df['관리번호'] == management_number]

    # SPEC 정보 가져오기
    spec_info = spec_df[spec_df['관리번호'] == management_number].iloc[0]
    usl = spec_info['USL']
    lsl = spec_info['LSL']
    target = spec_info['Target']

    # 측정값
    values = filtered_df['측정값'].values

    # 히스토그램 생성
    fig = px.histogram(
        filtered_df,
        x='측정값',
        nbins=20,
        marginal='box',
        title=f"측정값 분포 히스토그램 (관리번호: {management_number})"
    )

    # USL, LSL, Target 선 추가
    fig.add_vline(x=usl, line_dash="solid", line_color="red", annotation_text="USL")
    fig.add_vline(x=lsl, line_dash="solid", line_color="red", annotation_text="LSL")
    fig.add_vline(x=target, line_dash="dash", line_color="green", annotation_text="Target")

    # 그래프 레이아웃 설정
    fig.update_layout(
        xaxis_title="측정값",
        yaxis_title="빈도수",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def create_trend_analysis(df, spec_df, management_number):
    """시간에 따른 추세 분석 그래프를 생성하는 함수"""
    # 선택된 관리번호에 해당하는 데이터 필터링
    filtered_df = df[df['관리번호'] == management_number].copy()

    # 날짜별 집계
    filtered_df['Date'] = filtered_df['측정일자'].dt.date
    daily_stats = filtered_df.groupby('Date').agg(
        평균=('측정값', 'mean'),
        최대=('측정값', 'max'),
        최소=('측정값', 'min'),
        표준편차=('측정값', 'std')
    ).reset_index()

    # SPEC 정보 가져오기
    spec_info = spec_df[spec_df['관리번호'] == management_number].iloc[0]
    usl = spec_info['USL']
    lsl = spec_info['LSL']

    # 추세 그래프 생성
    fig = go.Figure()

    # 일별 평균 측정값 추가
    fig.add_trace(
        go.Scatter(
            x=daily_stats['Date'],
            y=daily_stats['평균'],
            mode='lines+markers',
            name='일별 평균',
            line=dict(color='blue')
        )
    )

    # 오차 범위 추가 (평균 ± 표준편차)
    fig.add_trace(
        go.Scatter(
            x=daily_stats['Date'],
            y=daily_stats['평균'] + daily_stats['표준편차'],
            mode='lines',
            name='평균 + 표준편차',
            line=dict(color='lightblue', dash='dash')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=daily_stats['Date'],
            y=daily_stats['평균'] - daily_stats['표준편차'],
            mode='lines',
            name='평균 - 표준편차',
            line=dict(color='lightblue', dash='dash'),
            fill='tonexty'  # 두 선 사이의 영역을 채움
        )
    )

    # USL, LSL 선 추가
    fig.add_trace(
        go.Scatter(
            x=[daily_stats['Date'].min(), daily_stats['Date'].max()],
            y=[usl, usl],
            mode='lines',
            name='USL',
            line=dict(color='red')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[daily_stats['Date'].min(), daily_stats['Date'].max()],
            y=[lsl, lsl],
            mode='lines',
            name='LSL',
            line=dict(color='red')
        )
    )

    # 그래프 레이아웃 설정
    fig.update_layout(
        title=f"일별 측정값 추세 분석 (관리번호: {management_number})",
        xaxis_title="날짜",
        yaxis_title="측정값",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def create_boxplot(df, spec_df, management_number):
    """관리번호별 박스플롯을 생성하는 함수"""
    # 선택된 관리번호에 해당하는 데이터 필터링
    filtered_df = df[df['관리번호'] == management_number].copy()

    # 날짜 형식 변환 (월-일)
    filtered_df['측정일자_요약'] = filtered_df['측정일자'].dt.strftime('%m-%d')

    # SPEC 정보 가져오기
    spec_info = spec_df[spec_df['관리번호'] == management_number].iloc[0]
    usl = spec_info['USL']
    lsl = spec_info['LSL']
    target = spec_info['Target']

    # 박스플롯 생성
    fig = px.box(
        filtered_df,
        x='측정일자_요약',
        y='측정값',
        title=f"일별 측정값 분포 (관리번호: {management_number})"
    )

    # USL, LSL, Target 선 추가
    fig.add_hline(y=usl, line_dash="solid", line_color="red", annotation_text="USL")
    fig.add_hline(y=lsl, line_dash="solid", line_color="red", annotation_text="LSL")
    fig.add_hline(y=target, line_dash="dash", line_color="green", annotation_text="Target")

    # 그래프 레이아웃 설정
    fig.update_layout(
        xaxis_title="측정일자",
        yaxis_title="측정값",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


# 메인 앱 로직
if uploaded_file is not None:
    try:
        # 데이터 로드
        df_toLGE, df_SPEC = load_data(uploaded_file)

        # 관리번호 목록
        management_numbers = df_toLGE['관리번호'].unique()

        # 필터링 옵션
        col1, col2 = st.columns(2)

        with col1:
            selected_management_number = st.selectbox(
                "관리번호 선택",
                options=management_numbers,
                key="management_number_select"
            )

        with col2:
            selected_chart_type = st.selectbox(
                "차트 유형",
                options=["개별값 관리도", "X-bar R 관리도"],
                index=0,
                key="chart_type_select"
            )

        # 선택된 관리번호 정보 표시
        if selected_management_number:
            # 해당 관리번호의 데이터 필터링
            filtered_df = df_toLGE[df_toLGE['관리번호'] == selected_management_number]
            spec_info = df_SPEC[df_SPEC['관리번호'] == selected_management_number].iloc[0]

            # 기본 정보 표시
            st.subheader("선택된 관리항목 정보")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("관리항목명", filtered_df['CTQ/P 관리항목명'].iloc[0])
            with col2:
                st.metric("Target", f"{spec_info['Target']:.3f}")
            with col3:
                st.metric("USL", f"{spec_info['USL']:.3f}")
            with col4:
                st.metric("LSL", f"{spec_info['LSL']:.3f}")

            # 공정능력지수 계산 및 표시
            capability = calculate_process_capability(df_toLGE, df_SPEC, selected_management_number)

            st.subheader("공정능력 분석")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("평균(Mean)", f"{capability['Mean']:.3f}")
                st.metric("표준편차", f"{capability['StdDev']:.3f}")

            with col2:
                st.metric("Cp", f"{capability['Cp']:.3f}")
                st.metric("Cpk", f"{capability['Cpk']:.3f}")

            with col3:
                st.metric("Pp", f"{capability['Pp']:.3f}")
                st.metric("Ppk", f"{capability['Ppk']:.3f}")

            with col4:
                st.metric("공정중심도(K)", f"{capability['K']:.3f}")
                cpk_status = "양호" if capability['Cpk'] >= 1.33 else "개선필요" if capability['Cpk'] >= 1.0 else "불량"
                st.metric("공정능력 판정", cpk_status)

            # 탭 생성
            tab1, tab2, tab3, tab4 = st.tabs(["관리도", "히스토그램", "추세 분석", "박스플롯"])

            with tab1:
                # 관리도 차트
                fig_control = create_control_chart(
                    df_toLGE,
                    df_SPEC,
                    selected_management_number,
                    chart_type='individual' if selected_chart_type == "개별값 관리도" else 'xbar-r'
                )
                st.plotly_chart(fig_control, use_container_width=True)

                # 관리도 해석
                st.subheader("관리도 해석")

                # SPEC 초과 확인
                filtered_df = df_toLGE[df_toLGE['관리번호'] == selected_management_number]
                spec_info = df_SPEC[df_SPEC['관리번호'] == selected_management_number].iloc[0]
                usl = spec_info['USL']
                lsl = spec_info['LSL']

                out_of_spec = filtered_df[(filtered_df['측정값'] > usl) | (filtered_df['측정값'] < lsl)]
                out_of_spec_percentage = (len(out_of_spec) / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("SPEC 초과 건수", f"{len(out_of_spec)} / {len(filtered_df)}")
                with col2:
                    st.metric("SPEC 초과율", f"{out_of_spec_percentage:.2f}%")

                if len(out_of_spec) > 0:
                    st.markdown("**SPEC 초과 데이터:**")
                    st.dataframe(out_of_spec[['측정일자', '측정값', 'CTQ/P 관리항목명']])

            with tab2:
                # 히스토그램
                fig_hist = create_histogram(df_toLGE, df_SPEC, selected_management_number)
                st.plotly_chart(fig_hist, use_container_width=True)

                # 분포 정보
                st.subheader("분포 특성")

                # 정규성 검정 (Shapiro-Wilk test)
                from scipy import stats

                values = filtered_df['측정값'].values
                shapiro_stat, shapiro_p = stats.shapiro(values)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Shapiro-Wilk 통계량", f"{shapiro_stat:.4f}")
                with col2:
                    st.metric("p-value", f"{shapiro_p:.4f}")

                if shapiro_p < 0.05:
                    st.warning("p-value가 0.05보다 작아 정규분포를 따르지 않는 것으로 판단됩니다.")
                else:
                    st.success("p-value가 0.05보다 커 정규분포를 따르는 것으로 판단됩니다.")

            with tab3:
                # 추세 분석
                fig_trend = create_trend_analysis(df_toLGE, df_SPEC, selected_management_number)
                st.plotly_chart(fig_trend, use_container_width=True)

                # 추세 분석 결과
                st.subheader("추세 분석 결과")

                # 날짜별 집계
                filtered_df['Date'] = filtered_df['측정일자'].dt.date
                daily_stats = filtered_df.groupby('Date').agg(
                    평균=('측정값', 'mean'),
                    최대=('측정값', 'max'),
                    최소=('측정값', 'min'),
                    표준편차=('측정값', 'std'),
                    개수=('측정값', 'count')
                ).reset_index()

                # 증가/감소 추세 확인
                from scipy import stats

                x = np.arange(len(daily_stats))
                y = daily_stats['평균'].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                trend_direction = "증가" if slope > 0 else "감소"
                significant_trend = p_value < 0.05

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("추세 방향", trend_direction)
                    st.metric("기울기", f"{slope:.4f}")
                with col2:
                    st.metric("R-squared", f"{r_value ** 2:.4f}")
                    st.metric("p-value", f"{p_value:.4f}")

                if significant_trend:
                    st.warning(f"통계적으로 유의한 {trend_direction} 추세가 감지되었습니다 (p < 0.05).")
                else:
                    st.success("통계적으로 유의한 추세가 없습니다.")

                # 일별 통계 데이터 표시
                st.subheader("일별 측정 통계")
                st.dataframe(daily_stats)

            with tab4:
                # 박스플롯
                fig_box = create_boxplot(df_toLGE, df_SPEC, selected_management_number)
                st.plotly_chart(fig_box, use_container_width=True)

                # 이상치 분석
                st.subheader("이상치 분석")

                # 일자별 이상치 확인
                filtered_df['측정일자_요약'] = filtered_df['측정일자'].dt.strftime('%m-%d')

                # IQR 방식으로 이상치 탐지
                Q1 = filtered_df['측정값'].quantile(0.25)
                Q3 = filtered_df['측정값'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = filtered_df[(filtered_df['측정값'] < lower_bound) | (filtered_df['측정값'] > upper_bound)]
                outlier_percentage = (len(outliers) / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("이상치 개수", f"{len(outliers)} / {len(filtered_df)}")
                with col2:
                    st.metric("이상치 비율", f"{outlier_percentage:.2f}%")

                if len(outliers) > 0:
                    st.markdown("**이상치 데이터:**")
                    st.dataframe(outliers[['측정일자', '측정값', 'CTQ/P 관리항목명']])

            # 데이터 내보내기 기능
            st.subheader("데이터 내보내기")

            # 선택된 관리번호의 데이터
            export_data = filtered_df[['측정일자', 'CTQ/P 관리항목명', '관리번호', '측정값']]

            # 엑셀 파일로 내보내기
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                export_data.to_excel(writer, sheet_name='Data', index=False)

                # 분석 결과 시트 추가
                analysis_result = pd.DataFrame({
                    'Metric': ['Mean', 'StdDev', 'USL', 'LSL', 'Target', 'Cp', 'Cpk', 'Pp', 'Ppk', 'K'],
                    'Value': [
                        capability['Mean'], capability['StdDev'],
                        capability['USL'], capability['LSL'], capability['Target'],
                        capability['Cp'], capability['Cpk'],
                        capability['Pp'], capability['Ppk'], capability['K']
                    ]
                })
                analysis_result.to_excel(writer, sheet_name='Analysis', index=False)

            output.seek(0)

            download_filename = f"품질분석_{selected_management_number}_{datetime.now().strftime('%Y%m%d')}.xlsx"
            st.download_button(
                label="Excel 파일 다운로드",
                data=output,
                file_name=download_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"오류 발생: {e}")
        st.exception(e)
else:
    # 파일이 업로드되지 않은 경우
    st.info("위에서 Excel 파일을 업로드해주세요.")

    # 예시 화면 또는 사용법 설명
    st.subheader("사용 방법")
    st.markdown("""
    1. Excel 파일을 업로드합니다. (필수 시트: 'toLGE', 'SPEC')
    2. 관리번호를 선택하여 해당 항목의 품질 분석 결과를 확인합니다.
    3. 관리도, 히스토그램, 추세 분석, 박스플롯 탭을 통해 다양한 분석 결과를 볼 수 있습니다.
    4. 필요한 경우 데이터와 분석 결과를 Excel 파일로 내보낼 수 있습니다.
    """)

    st.markdown("""
    ### 주요 기능

    - **관리도**: 시간에 따른 측정값 추이와 관리 한계선 표시
    - **히스토그램**: 측정값의 분포 및 정규성 검정
    - **추세 분석**: 일별 통계 및 추세 분석
    - **박스플롯**: 일별 측정값 분포 및 이상치 분석
    """)

# 푸터
st.markdown("---")
st.markdown("© 2025 품질관리 대시보드 | 개발: Your Company Name")
st.markdown("문의: [quality@example.com](mailto:quality@example.com)")
