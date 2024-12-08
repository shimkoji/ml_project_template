import os


def generate_tree(path, max_depth=3, ignore_patterns=None):
    if ignore_patterns is None:
        ignore_patterns = [".git", "__pycache__", "uv.lock", ".python-version"]

    tree_string = ""
    for root, dirs, files in os.walk(path):
        depth = root.replace(path, "").count(os.sep)
        if depth > max_depth:
            continue

        indent = "  " * depth
        basename = os.path.basename(root)
        if basename in ignore_patterns:
            continue

        tree_string += f"{indent}├── {basename}\n"

        for file in files:
            if file in ignore_patterns:
                continue
            tree_string += f"{indent}│   └── {file}\n"

        # ディレクトリはフィルタリングした後にサブディレクトリへ進む
        dirs[:] = [d for d in dirs if d not in ignore_patterns]
        for dir in dirs:
            tree_string += f"{indent}│   ├── {dir}\n"

    return tree_string


if __name__ == "__main__":
    tree_output = generate_tree(".")
    with open("README.md", "w") as f:
        f.write(tree_output)
