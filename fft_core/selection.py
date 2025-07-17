import logging

logger = logging.getLogger(__name__)

fft_functions = {}
_duplicates_names = {}

def register_fft(func=None, *, name=None):
    """
    Decorator to register an FFT implementation.
    Usage:
      @register_fft
      def fft(x): ...
    or
      @register_fft(name="superfft")
      def my_fft(x): ...
    """
    def _register(f):
        key = name or f.__name__
        if key in fft_functions:
            i = _duplicates_names.get(key, 0) + 1
            _duplicates_names[key] = i
            key = f"{key}_{i}"
            logger.warning(f"Duplicate FFT name detected—registering as '{key}'")
            
        fft_functions[key] = f
        logger.info(f"Registered FFT implementation: '{key}'")
        return f

    # support both forms
    return _register(func) if func else _register