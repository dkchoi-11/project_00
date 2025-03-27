import pandas as pd

# 문자열 변환용 함수
def clean_string(s):
    return str(s).strip().replace("/", "-")

# 결과 저장 함수
def save_to_excel(df: pd.DataFrame) -> None:

    # 중복 제거 후 고유값 추출
    first_company = clean_string(df['1차 업체명'].unique()[0])
    region = clean_string(df['지역명'].unique()[0])
    second_company = clean_string(df['2차업체명'].unique()[0])
    model = clean_string(df['모델명'].unique()[0])
    part_name = clean_string(df['부품명'].unique()[0])

    # 날짜 형식 변환 및 정렬
    df["측정일자"] = pd.to_datetime(df["측정일자"])
    start_date = df["측정일자"].min().strftime("%Y%m%d")
    end_date = df["측정일자"].max().strftime("%Y%m%d")

    # 파일명 생성
    filename = f"CTQ_{first_company}_{region}_{second_company}_{model}_{part_name}_{start_date}_{end_date}.xlsx"

    # 엑셀 저장 (시트 이름 toLGE)
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='toLGE', index=False)

    print(f"파일이 저장되었습니다: {filename}")

