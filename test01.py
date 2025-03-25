import pandas as pd
from openpyxl import load_workbook
from datetime import datetime

# 날짜행 탐지 함수
def find_date_row(df, date_start_col=5):
    def is_date(x):
        if isinstance(x, (pd.Timestamp, datetime)):
            return True
        if isinstance(x, str):
            try:
                pd.to_datetime(x)
                return True
            except:
                return False
        return False

    for i in range(len(df)):
        row = df.iloc[i, date_start_col:]
        if row.apply(is_date).sum() >= 3:
            return i
    raise ValueError("날짜가 포함된 행을 찾을 수 없습니다.")

# 날짜 열 인덱스 → 날짜 매핑
def get_date_mapping(df, date_row_index, date_start_col=5):
    date_row = df.iloc[date_row_index, date_start_col:]
    dates = pd.to_datetime(date_row, errors='coerce')

    mapping = {}
    for col_idx, date in zip(range(date_start_col, len(df.columns)), dates):
        if pd.notna(date):
            mapping[col_idx] = date
    return mapping

# CTQ별 측정값 추출
def extract_measurement_data(df, date_mapping, date_row_index, point_col=1, ctq_col=2):
    results = []

    # 측정POINT 위치 수집
    point_indices = []
    for i in range(date_row_index, len(df)):
        if df.iloc[i, point_col] == '측정POINT' and pd.notna(df.iloc[i, ctq_col]):
            point_indices.append((i, df.iloc[i, ctq_col]))

    # CTQ 구간별 측정값 수집
    for idx, (start_idx, ctq_name) in enumerate(point_indices):
        # 다음 CTQ까지 or 마지막 유효 행까지
        if idx + 1 < len(point_indices):
            end_idx = point_indices[idx + 1][0]
        else:
            end_idx = start_idx + 1
            while end_idx < len(df):
                if df.iloc[end_idx, min(date_mapping.keys()):].notna().sum() > 0:
                    end_idx += 1
                else:
                    break

        # start_idx 부터 포함
        for i in range(start_idx, end_idx):
            for col in date_mapping:
                value = df.iloc[i, col]
                if pd.notna(value):
                    results.append({
                        'Date': date_mapping[col],
                        '관리항목명': ctq_name,
                        '측정값': value
                    })

    return pd.DataFrame(results)

# 전체 실행 함수
def process_excel_and_save(input_path, output_path, sheet_name='SUB', save_sheet='변환'):
    df = pd.read_excel(input_path, sheet_name=sheet_name, header=None)
    date_row_idx = find_date_row(df)
    date_map = get_date_mapping(df, date_row_idx)
    df_result = extract_measurement_data(df, date_map, date_row_idx)

    # 정렬
    df_result_sorted = df_result.sort_values(by=["Date", "관리항목명"]).reset_index(drop=True)

    # 저장
    wb = load_workbook(input_path)
    if save_sheet in wb.sheetnames:
        del wb[save_sheet]
    ws_new = wb.create_sheet(save_sheet)
    ws_new.append(["Date", "관리항목명", "측정값"])
    for row in df_result_sorted.itertuples(index=False):
        ws_new.append([row.Date, row.관리항목명, row.측정값])
    wb.save(output_path)
    print(f"✅ 변환 완료: {output_path}")

process_excel_and_save(
    input_path="test00.xlsx",
    output_path="test00_변환완료_함수버전.xlsx"
)