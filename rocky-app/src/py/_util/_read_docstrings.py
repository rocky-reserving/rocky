import ast


def get_docstrings(filename):
    with open(filename, "r") as file:
        module_node = ast.parse(file.read())

    return module_node

    # docstrings = {}

    # for node in module_node.body:
    #     if isinstance(node, ast.Assign):
    #         if isinstance(node.value, ast.Str):
    #             docstrings[filename] = [node.value.s]
    #     if isinstance(node, ast.FunctionDef):
    #         if ast.get_docstring(node) is not None:
    #             docstrings[node.name] = [ast.get_docstring(node)]
    #     if isinstance(node, ast.ClassDef):
    #         if ast.get_docstring(node) is not None:
    #             docstrings[node.name] = [ast.get_docstring(node)]

    # return docstrings
