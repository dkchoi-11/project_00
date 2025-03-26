import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Union
from datetime import datetime
from io import BytesIO

# 날짜 형식이 가장 많이 들어있는 열의 인덱스를 찾는 함수
def find_date_start_col(df: pd.DataFrame, sample_row_count: int = 10) -> int:
    def is_date(x: Union[str, datetime]) -> bool:
        try:
            pd.to_datetime(x)
            return True
        except:
            return False

    date_counts = [
        sum(df.iloc[:sample_row_count, col_idx].apply(is_date))
        for col_idx in range(df.shape[1])
    ]
    best_col = date_counts.index(max(date_counts))

    if date_counts[best_col] == 0:
        raise ValueError("No date column found")

    return best_col

# 날짜가 포함된 첫 번째 행의 인덱스를 찾는 함수
def find_date_row(df: pd.DataFrame, date_start_col: Optional[int] = None,
                  min_date_count: int = 1, max_row_check: int = 20) -> int:
    if date_start_col is None:
        date_start_col = find_date_start_col(df)

    def is_date(x: Union[str, datetime, pd.Timestamp]) -> bool:
        if isinstance(x, (pd.Timestamp, datetime)):
            return True
        if isinstance(x, str):
            try:
                pd.to_datetime(x)
                return True
            except:
                return False
        return False

    for i in range(min(max_row_check, len(df))):
        row = df.iloc[i, date_start_col:]
        if row.apply(is_date).sum() >= min_date_count:
            return i

    raise ValueError("No date row found")

# 날짜가 들어 있는 셀들의 열 인덱스를 실제 날짜 값과 매핑하는 함수
def get_date_mapping(df: pd.DataFrame, date_row_index: int,
                     date_start_col: Optional[int] = None) -> Dict[int, pd.Timestamp]:
    if date_start_col is None:
        date_start_col = find_date_start_col(df)

    date_row = df.iloc[date_row_index, date_start_col:]
    dates = pd.to_datetime(date_row, errors='coerce')

    mapping = {}
    for col_idx, date in zip(range(date_start_col, len(df.columns)), dates):
        if pd.notna(date):
            mapping[col_idx] = date

    return mapping

# 측정 데이터를 추출하는 함수
def extract_measurement_data(
        df: pd.DataFrame,
        info_dict: Dict[str, str],
        date_mapping: Dict[int, pd.Timestamp],
        date_row_index: int,
        search_cols: List[int] = list(range(0, 11))
) -> pd.DataFrame:
    results = []
    search_cols = [col for col in search_cols if col < df.shape[1]]

    point_indices = []
    for i in range(date_row_index, len(df)):
        for col in search_cols:
            if col >= df.shape[1]:
                continue
            cell_value = df.iloc[i, col]
            if str(cell_value).strip() == '측정POINT':
                ctq_col = col + 1
                if ctq_col < df.shape[1]:
                    ctq_name = df.iloc[i, ctq_col]
                    if pd.notna(ctq_name):
                        point_indices.append((i, ctq_name))
                break

    for idx, (start_idx, ctq_name) in enumerate(point_indices):
        if idx + 1 < len(point_indices):
            end_idx = point_indices[idx + 1][0]
        else:
            end_idx = start_idx + 1
            while end_idx < len(df):
                if df.iloc[end_idx, min(date_mapping.keys()):].notna().sum() > 0:
                    end_idx += 1
                else:
                    break

        for i in range(start_idx, end_idx):
            for col in date_mapping:
                if col < df.shape[1] and i < df.shape[0]:
                    value = df.iloc[i, col]
                    if pd.notna(value):
                        results.append({
                            '1차 업체명': str(info_dict.get("1차 업체명", "")).strip(),
                            '지역명': str(info_dict.get("지역명", "")).strip(),
                            '2차업체명': str(info_dict.get("2차업체명", "")).strip(),
                            '모델명': str(info_dict.get("모델명", "")).strip(),
                            '측정자': str(info_dict.get("측정자", "")).strip(),
                            '측정장비': str(info_dict.get("측정장비", "")).strip(),
                            '부품명': str(info_dict.get("부품명", "")).strip(),
                            'CTQ/P 관리항목명': str(ctq_name).strip(),
                            '측정일자': date_mapping[col],
                            '측정값': value,
                            'Part No': str(info_dict.get("Part No", "")).strip()
                        })

    return pd.DataFrame(results)

# 관리번호를 매핑하는 함수
def add_management_code(results_df: pd.DataFrame, master_key_df: pd.DataFrame) -> pd.DataFrame:
    merge_keys = ["1차 업체명", "지역명", "2차업체명", "모델명", "부품명", "CTQ/P 관리항목명"]

    master_key_df_renamed = master_key_df.rename(columns={
        "부품": "부품명",
        "공정CTQ/CTP 관리 항목명": "CTQ/P 관리항목명",
        "2차 업체명": "2차업체명"
    })

    master_key_df_renamed = master_key_df_renamed.loc[:, ["관리번호"] + merge_keys]

    for key in merge_keys:
        results_df[key] = results_df[key].astype(str).str.strip()
        master_key_df_renamed[key] = master_key_df_renamed[key].astype(str).str.strip()

    merged_df = pd.merge(results_df, master_key_df_renamed, on=merge_keys, how='left')

    # 관리번호를 첫 번째 열로 이동
    if "관리번호" in merged_df.columns:
        cols = merged_df.columns.tolist()
        cols.remove("관리번호")
        merged_df = merged_df[["관리번호"] + cols]

    return merged_df

# 전체 프로세스를 실행하는 함수
def transform_data(
        input_file: BytesIO,
        master_file: BytesIO,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        search_cols: List[int] = list(range(0, 11))
) -> pd.DataFrame:
    info_df = pd.read_excel(input_file, sheet_name="Information")
    info_dict = info_df.set_index("Contents")['Value'].to_dict()

    df = pd.read_excel(input_file, sheet_name=info_dict["Data_sheet"], header=None)
    master_df = pd.read_excel(master_file, sheet_name="Master")

    date_row_idx = find_date_row(df)
    date_map = get_date_mapping(df, date_row_idx)

    df_result = extract_measurement_data(df, info_dict, date_map, date_row_idx, search_cols)
    df_result = add_management_code(df_result, master_df)

    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df_result = df_result[
            (df_result["측정일자"] >= start_date) &
            (df_result["측정일자"] <= end_date)
        ]

    df_result_sorted = df_result.sort_values(by=["측정일자", "CTQ/P 관리항목명"]).reset_index(drop=True)
    return df_result_sorted

# Streamlit 앱 실행
st.title("측정값 변환 도구")

input_file = st.file_uploader("📄 측정값 Excel 파일 업로드", type=["xlsx"])
master_file = st.file_uploader("📄 마스터 키 Excel 파일 업로드", type=["xlsx"])
start_date = st.date_input("시작 날짜", value=None)
end_date = st.date_input("종료 날짜", value=None)

if input_file and master_file and start_date and end_date:
    try:
        df_result = transform_data(
            input_file=input_file,
            master_file=master_file,
            start_date=start_date,
            end_date=end_date
        )

        st.success("✅ 변환 완료!")
        st.dataframe(df_result)

        towrite = BytesIO()
        with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
            df_result.to_excel(writer, index=False, sheet_name="변환")
        towrite.seek(0)

        st.download_button(
            label="📥 결과 다운로드",
            data=towrite,
            file_name="변환결과.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
