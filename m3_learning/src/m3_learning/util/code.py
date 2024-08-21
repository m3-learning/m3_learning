from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
import inspect

def print_code(func):
    """
    print_code A function to print the source code of a function with syntax highlighting and line numbers.

    Args:
        func (obj): a python function
    """    
    
    # Fetch source lines and starting line number
    source_lines, starting_line_number = inspect.getsourcelines(func)
    source_code = "".join(source_lines)

    # Perform syntax highlighting using Pygments
    highlighted_code = highlight(source_code, PythonLexer(), TerminalFormatter())

    # Print highlighted code with line numbers
    for i, line in enumerate(highlighted_code.split("\n"), start=starting_line_number):
        print(f"{i}: {line}")