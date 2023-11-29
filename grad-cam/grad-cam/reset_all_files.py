import os

def delete_files_in_folder(folder_path):
    # ensure the folder path is valid
    if not os.path.exists(folder_path):
        print(f"The specified folder path '{folder_path}' does not exist.")
        return

    # get a list of files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    if (files == []):
        print("No file processed.")

    # delete each file
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        try:
            os.remove(file_path)
            print(f"Deleted: {file_name}")
        except Exception as e:
            print(f"Error deleting {file_name}: {e}")

delete_files_in_folder('grad-cam/data/images')
delete_files_in_folder('grad-cam/results')