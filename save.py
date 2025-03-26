import pandas as pd

# 결과 저장 함수
def save_to_excel(df: pd.DataFrame, output_path: str, sheet_name: str = '변환') -> None:
    columns = df.columns.tolist()
    if "관리번호" in columns:
        columns.remove("관리번호")
        columns = ["관리번호"] + columns
        df = df[columns]

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
