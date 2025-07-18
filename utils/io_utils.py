"""Utility functions for I/O tasks."""


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
