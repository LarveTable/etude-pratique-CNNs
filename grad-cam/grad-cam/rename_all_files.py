import os

def rename_files(folder_path):
    # Ensure the folder path is valid
    if not os.path.exists(folder_path):
        print(f"The specified folder path '{folder_path}' does not exist.")
        return

    # Get a list of files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and not f.startswith('.')]
    
    # Sort the files
    files.sort()

    # Rename the files in numerical order
    for index, old_name in enumerate(files, start=1):
        # Get the file extension
        _, extension = os.path.splitext(old_name)

        # Create the new file name with a numerical prefix
        new_name = f"{index:03d}{extension}"

        # Build the full paths for old and new names
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {old_name} -> {new_name}")
