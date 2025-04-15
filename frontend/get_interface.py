import argparse
import ast
import json
def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract function and method information from a Python code file.')
    parser.add_argument('file', help='Path to the Python code file')
    return parser.parse_args()

class ExtractorVisitor(ast.NodeVisitor):
    def __init__(self):
        self.current_class = None
        self.functions = []

    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        for child in node.body:
            self.visit(child)
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node):
        self._process_function(node, is_async=True)

    def _process_function(self, node, is_async=False):
        func_info = {
            'name': node.name,
            'class': self.current_class,
            'async': is_async,
            'args': self._parse_arguments(node.args),
            'returns': self._parse_annotation(node.returns),
            'docstring': ast.get_docstring(node)
        }
        self.functions.append(func_info)
        self.generic_visit(node)

    def _parse_arguments(self, args_node):
        parameters = []
        pos_defaults = args_node.defaults
        num_pos = len(args_node.args)
        num_pos_defaults = len(pos_defaults)

        for i, arg in enumerate(args_node.args):
            default = pos_defaults[i - (num_pos - num_pos_defaults)] if i >= (num_pos - num_pos_defaults) else None
            parameters.append({
                'name': arg.arg,
                'type': self._parse_annotation(arg.annotation),
                'default': self._parse_default(default),
                'kind': 'positional'
            })

        if args_node.vararg:
            parameters.append({
                'name': args_node.vararg.arg,
                'type': self._parse_annotation(args_node.vararg.annotation),
                'default': None,
                'kind': 'varargs'
            })

        for i, arg in enumerate(args_node.kwonlyargs):
            default = args_node.kw_defaults[i] if i < len(args_node.kw_defaults) else None
            parameters.append({
                'name': arg.arg,
                'type': self._parse_annotation(arg.annotation),
                'default': self._parse_default(default),
                'kind': 'keyword_only'
            })

        if args_node.kwarg:
            parameters.append({
                'name': args_node.kwarg.arg,
                'type': self._parse_annotation(args_node.kwarg.annotation),
                'default': None,
                'kind': 'kwargs'
            })

        return parameters

    def _parse_annotation(self, node):
        if node is None:
            return None
        try:
            return ast.unparse(node)
        except AttributeError:
            return 'Annotation (unavailable in this Python version)'

    def _parse_default(self, node):
        if node is None:
            return None
        try:
            return ast.unparse(node)
        except AttributeError:
            return 'Default (unavailable in this Python version)'


def format_output(functions):
    output_data = []
    
    for func in functions:
        name = f"{func['class']}.{func['name']}" if func['class'] else func['name']
        if func['async']:
            name = f"async {name}"
        
        function_data = {
            "Function": name,
            "Parameters": [],
            "Returns": func['returns'] or 'None',
  #          "Docstring": func['docstring'] or 'No docstring'
        }
        
        for param in func['args']:
            param_desc = {
                "name": param['name'],
          #      "type": param['type'] or 'No type',
           #     "default": str(param['default']) if param['default'] is not None else 'No default',
            #    "kind": param['kind']
            }
            function_data["Parameters"].append(param_desc)
        
        output_data.append(function_data)
    
    # Write to JSON file
    with open('frontend/interface.json', 'w') as json_file:
        json.dump(output_data, json_file, indent=4)
    
    return output_data

def main():
    args = parse_arguments()
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            code = f.read()
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found.")
        return

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"Syntax error in file: {e}")
        return

    visitor = ExtractorVisitor()
    visitor.visit(tree)
    format_output(visitor.functions)

if __name__ == "__main__":
    main()
