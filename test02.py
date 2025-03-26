import pandas as pd
import logging
from typing import Optional, List, Dict, Union
from openpyxl import load_workbook
from datetime import datetime

# 로깅 설정: 로그 메시지를 INFO 레벨로 설정하고 출력 형식을 지정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 날짜 형식이 가장 많이 들어있는 열의 인덱스를 찾는 함수
def find_date_start_col(df: pd.DataFrame, sample_row_count: int = 10) -> int:
    def is_date(x: Union[str, datetime]) -> bool:
        try:
            pd.to_datetime(x)
            return True
        except:
            return False

    try:
        # 각 열마다 상위 sample_row_count 행에 날짜처럼 보이는 값 개수 측정
        date_counts = [
            sum(df.iloc[:sample_row_count, col_idx].apply(is_date))
            for col_idx in range(df.shape[1])
        ]
        best_col = date_counts.index(max(date_counts))

        if date_counts[best_col] == 0:
            raise ValueError("No date column found")

        return best_col

    except Exception as e:
        logger.error(f"Error finding date column: {e}")
        raise

# 날짜가 포함된 첫 번째 행의 인덱스를 찾는 함수
def find_date_row(df: pd.DataFrame, date_start_col: Optional[int] = None,
                  min_date_count: int = 1, max_row_check: int = 20) -> int:
    try:
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

        # 상단 max_row_check 행까지 탐색하면서 날짜가 min_date_count 개 이상 있는 행을 찾음
        for i in range(min(max_row_check, len(df))):
            row = df.iloc[i, date_start_col:]
            if row.apply(is_date).sum() >= min_date_count:
                return i

        raise ValueError("No date row found")

    except Exception as e:
        logger.error(f"Error finding date row: {e}")
        raise

# 날짜가 들어 있는 셀들의 열 인덱스를 실제 날짜 값과 매핑하는 함수
def get_date_mapping(df: pd.DataFrame, date_row_index: int,
                     date_start_col: Optional[int] = None) -> Dict[int, pd.Timestamp]:
    try:
        if date_start_col is None:
            date_start_col = find_date_start_col(df)

        # 날짜 행에서 유효한 날짜만 추출
        date_row = df.iloc[date_row_index, date_start_col:]
        dates = pd.to_datetime(date_row, errors='coerce')

        mapping = {}
        for col_idx, date in zip(range(date_start_col, len(df.columns)), dates):
            if pd.notna(date):
                mapping[col_idx] = date

        return mapping

    except Exception as e:
        logger.error(f"Error creating date mapping: {e}")
        raise

# 측정 데이터를 추출하는 함수
def extract_measurement_data(
        df: pd.DataFrame,
        info_dict: Dict[str, str],
        date_mapping: Dict[int, pd.Timestamp],
        date_row_index: int,
        search_cols: List[int] = list(range(0, 11))
) -> pd.DataFrame:
    """
    시트에서 측정 데이터를 추출하고 info_dict의 정보를 병합한 결과를 반환합니다.
    """
    try:
        results = []

        # 실제 열 범위보다 큰 인덱스를 제거
        search_cols = [col for col in search_cols if col < df.shape[1]]

        # '측정POINT'가 포함된 행과 CTQ명을 찾음
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
                    break  # 하나 찾았으면 다음 행으로 넘어감

        # CTQ별 측정값 수집
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

    except Exception as e:
        logger.error(f"Error extracting measurement data: {e}")
        raise


# 관리번호를 매핑하는 함수
def add_management_code(results_df: pd.DataFrame, master_key_df: pd.DataFrame) -> pd.DataFrame:
    try:
        merge_keys = ["1차 업체명", "지역명", "2차업체명", "모델명", "부품명", "CTQ/P 관리항목명"]

        # 마스터 키의 컬럼명을 결과 데이터프레임과 일치하도록 변경
        master_key_df_renamed = master_key_df.rename(columns={
            "부품": "부품명",
            "공정CTQ/CTP 관리 항목명": "CTQ/P 관리항목명",
            "2차 업체명": "2차업체명"
        })

        # 병합 대상 열만 추출
        master_key_df_renamed = master_key_df_renamed.loc[:, merge_keys + ["관리번호"]]

        # 문자열 공백 제거 후 병합
        for key in merge_keys:
            results_df[key] = results_df[key].astype(str).str.strip()
            master_key_df_renamed[key] = master_key_df_renamed[key].astype(str).str.strip()

        # 관리번호 병합
        merged_df = pd.merge(results_df, master_key_df_renamed, on=merge_keys, how='left')

        return merged_df

    except Exception as e:
        logger.error(f"관리번호 매핑 중 오류 발생: {e}")
        raise

# 전체 프로세스를 실행하는 함수
def process_excel_and_save(
        input_path: str,
        output_path: str,
        master_path: str,
        sheet_name: str = 'SUB',
        save_sheet: str = '변환',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        search_cols: List[int] = list(range(0, 11))
) -> None:
    try:
        # Information 시트에서 메타 정보 읽기
        info_df = pd.read_excel(input_path, sheet_name="Information")
        info_dict = info_df.set_index("Contents")['Value'].to_dict()

        # 데이터 시트 및 마스터 시트 읽기
        df = pd.read_excel(input_path, sheet_name=info_dict["Data_sheet"], header=None)
        master_df = pd.read_excel(master_path, sheet_name="Master")

        # 날짜 위치 찾기
        date_row_idx = find_date_row(df)
        date_map = get_date_mapping(df, date_row_idx)

        # 측정 데이터 추출
        df_result = extract_measurement_data(df, info_dict, date_map, date_row_idx, search_cols)

        # 관리번호 매핑
        df_result = add_management_code(df_result, master_df)

        # 날짜 필터 적용
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            df_result = df_result[
                (df_result["측정일자"] >= start_date) &
                (df_result["측정일자"] <= end_date)
            ]

        # 정렬
        df_result_sorted = df_result.sort_values(by=["측정일자", "CTQ/P 관리항목명"]).reset_index(drop=True)

        # 관리번호 열을 첫 번째로 이동
        columns = df_result_sorted.columns.tolist()
        if "관리번호" in columns:
            columns.remove("관리번호")
            columns = ["관리번호"] + columns
            df_result_sorted = df_result_sorted[columns]

        # 결과 저장
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_result_sorted.to_excel(writer, sheet_name=save_sheet, index=False)

        logger.info(f"✅ 변환 완료: {output_path}")

    except FileNotFoundError:
        logger.error(f"파일을 찾을 수 없습니다: {input_path}")
    except PermissionError:
        logger.error(f"파일에 접근 권한이 없습니다: {input_path}")
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")

# 메인 실행 함수
def main():
    try:
        start = input("시작 날짜를 입력하세요 (예: 2024-01-06): ")
        end = input("종료 날짜를 입력하세요 (예: 2024-01-15): ")

        process_excel_and_save(
            input_path="test00.xlsx",
            output_path="test00_변환완료_함수버전.xlsx",
            master_path="LGE_Master_희성_뉴옵.xlsx",
            start_date=start,
            end_date=end
        )
    except KeyboardInterrupt:
        logger.info("작업이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"예기치 않은 오류 발생: {e}")

if __name__ == "__main__":
    main()
