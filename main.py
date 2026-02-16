from src.data.data_utils import data_checks

csv_path = "./Dataset/labels1.csv"
images_folder = "./Dataset/CTs"

# check df
clean_df = data_checks(csv_path, images_folder)

