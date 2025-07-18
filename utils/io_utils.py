"""Utility functions for I/O tasks like conditional and colored printing."""

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    print("WARNING: colorama is not installed. Colorized output will not be available.")
    Fore = Style = None

def qprint(output="", quiet=False, **kwargs):
    """
    Print the output unless the quiet flag is set to True.

    Parameters:
        output (str): The message to print. Defaults to an empty string.
        quiet (bool): If set to True, suppresses printing. Defaults to False.
        **kwargs: Additional keyword arguments passed to the built-in print().
    """
    if not quiet:
        print(output, **kwargs)

def colored_print(text, color=None, print_func=qprint, **kwargs):
    """
    Print text with optional color using a customizable print function.

    Parameters:
        text (str): The message to print.
        color (str, optional): The color name to use for text (case-insensitive).
            Supported colors (from colorama.Fore, if available):
                - BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET

        print_func (callable, optional): A print-like function to use (default: qprint).
            Should accept a `quiet` keyword argument for conditional printing.
        **kwargs: Additional keyword arguments passed to the print function.

    Examples:
        colored_print("Success!", color="green")
        colored_print("Warning!", color="yellow", quiet=True)
        colored_print("Raw print", print_func=print)

    Note:
        Make sure colorama is installed and initialized for full color support,
        especially on Windows terminals.
    """
    if Fore is None or Style is None:
        print_func(text, **kwargs)
        return

    color_code = getattr(Fore, color.upper(), None) if color else None
    if color_code is None:
        print_func(text, **kwargs)
        return

    colored_text = f"{color_code}{text}{Style.RESET_ALL}"
    print_func(colored_text, **kwargs)

# alias for brevity
cprint = colored_print

if __name__ == "__main__":
    colored_print("Hello, World!", color="red")
    cprint("Success!", color="green")