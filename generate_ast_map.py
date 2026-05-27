import ast
import os

def parse_file(filepath):
    with open(filepath, "r") as f:
        node = ast.parse(f.read())
    
    summary = {"classes": [], "functions": []}
    
    for item in node.body:
        if isinstance(item, ast.ClassDef):
            methods = [n.name for n in item.body if isinstance(n, ast.FunctionDef)]
            summary["classes"].append({"name": item.name, "methods": methods})
        elif isinstance(item, ast.FunctionDef):
            summary["functions"].append(item.name)
            
    return summary

def generate_project_map(root_dirs, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = ["venv", ".git", "__pycache__", ".continue"]
        
    project_map = {}
    
    for root_dir in root_dirs:
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    try:
                        project_map[full_path] = parse_file(full_path)
                    except Exception as e:
                        project_map[full_path] = f"Error parsing: {e}"
                        
    return project_map

def format_map(project_map):
    output = "# Project Symbol Map (AST Summary)\n\n"
    
    # Sort files for consistent output
    for filepath in sorted(project_map.keys()):
        output += f"## `{filepath}`\n"
        data = project_map[filepath]
        
        if isinstance(data, str):
            output += f"**{data}**\n\n"
            continue
            
        if data["classes"]:
            output += "### Classes\n"
            for cls in data["classes"]:
                output += f"- **{cls['name']}**\n"
                for method in cls["methods"]:
                    output += f"  - `{method}()`\n"
            output += "\n"
            
        if data["functions"]:
            output += "### Functions\n"
            for func in data["functions"]:
                output += f"- `{func}()`\n"
            output += "\n"
            
    return output

if __name__ == "__main__":
    dirs_to_scan = ["src", "tests", "."]
    pmap = generate_project_map(dirs_to_scan)
    formatted = format_map(pmap)
    
    with open("PROJECT_MAP.md", "w") as f:
        f.write(formatted)
    print("Project map generated in PROJECT_MAP.md")
