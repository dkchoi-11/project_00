import pandas as pd

input_path = r'C:\Python\Projects\project_00\LGE Master_희성_뉴옵.xlsx'


def get_information(input_path):
    info_df = pd.read_excel(input_path,sheet_name="Information")
    info_dict = info_df.set_index("Contents")["Value"].to_dict()

    data_sheet = info_dict.get("Data_sheet")
    supplier_1 = info_dict.get("1st_Supplier")
    region = info_dict.get("Region")
    supplier_2 = info_dict.get("2nd_Supplier")
    model = info_dict.get("Model")
    inspector = info_dict.get("Inspector")
    equipment = info_dict.get("equipment")
    part_name = info_dict.get("Part_Name")
    part_no = info_dict.get("Part_No")

    return data_sheet, supplier_1, region, supplier_2, model, inspector, equipment, part_name, part_no


