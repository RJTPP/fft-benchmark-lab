"""Utility functions for I/O tasks."""

from colorama import Fore, Style

def qprint(output="", quiet=False, **kwargs):
    """
    Prints the output unless the quiet flag is set to True.

    Parameters:
        output (str): The message to print. Defaults to an empty string.
        quiet (bool): If set to True, suppresses printing. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the print function.
    """

    if not quiet:
        print(output, **kwargs)


def colored_print(text, color=None, print_func=qprint, **kwargs):
    """
    Print text with optional color using a customizable print function.

    Parameters:
        text (str): The message to print.
        color (str, optional): The name of the color to use for text. Case-insensitive.
            If None or invalid, prints without color.

            Supported colors (from colorama.Fore):
                - BLACK
                - RED
                - GREEN
                - YELLOW
                - BLUE
                - MAGENTA
                - CYAN
                - WHITE
                - RESET

        print_func (callable, optional): A print-like function to use (default: qprint, from this module).
            It should accept a `quiet` keyword argument if you want to suppress output conditionally.
        **kwargs: Additional keyword arguments passed to the print function.

    Example:
        colored_print("Success!", color="green")
        colored_print("Silent warning", color="yellow", quiet=True)
        colored_print("Info", print_func=print)

    Note:
        Ensure `colorama.init()` has been called before using this function for proper color support,
        especially on Windows terminals.
    """
    color_code = getattr(Fore, color.upper(), None) if color else None
    if color_code is None:
        print_func(text, **kwargs)
        return
    colored_text = f"{color_code}{text}{Style.RESET_ALL}"
    print_func(colored_text, **kwargs)
    
    
if __name__ == "__main__":
    colored_print("Hello, World!", color="red")