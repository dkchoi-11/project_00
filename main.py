from transform import transform_data
from save import save_to_excel

# 메인 실행 함수
def main():
    try:
        start = input("시작 날짜를 입력하세요 (예: 2024-01-06): ")
        end = input("종료 날짜를 입력하세요 (예: 2024-01-15): ")

        result_df = transform_data(
            input_path="test00.xlsx",
            master_path="LGE_Master_희성_뉴옵.xlsx",
            start_date=start,
            end_date=end
        )

        save_to_excel(result_df, output_path="test00_변환완료_함수버전.xlsx", sheet_name="변환")

    except KeyboardInterrupt:
        print("작업이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"예기치 않은 오류 발생: {e}")

if __name__ == "__main__":
    main()