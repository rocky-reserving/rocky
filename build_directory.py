import os


def generate_file_structure_map(directory, output_file, skip_paths=None):
    with open(output_file, "w") as file:
        file.write(directory + "\n")
        traverse_directory(directory, file, "", skip_paths)


def traverse_directory(directory, file, indent, skip_paths):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if skip_paths and any(skip_path in item_path for skip_path in skip_paths):
            continue

        if os.path.isdir(item_path):
            file.write(indent + "|--\\" + item + "\\\n")
            traverse_directory(item_path, file, indent + "|  ", skip_paths)
        else:
            file.write(indent + "|--" + item + "\n")


# Example usage:
input_directory = r"C:\Users\AndyW\OneDrive\Documents\GitHub\rocky"
output_file = "rocky_structure_map.txt"
skip_paths = [
    ".gitignore",
    "node_modules",
    "__pycache__",
    ".vscode",
    "conda",
    "data",
    "instance",
    "migrations",
    "model_outputs",
    "old",
    "papers",
    "git",
    "testing",
]
generate_file_structure_map(input_directory, output_file, skip_paths)
