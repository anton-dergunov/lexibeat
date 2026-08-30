try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version  # For Python <3.8

try:
    __version__ = version("chatterbox-tts")
except PackageNotFoundError:
    # The Space distribution vendors the official source instead of installing
    # its package so ZeroGPU can supply its own compatible Torch build.
    __version__ = "0.1.7+vendored"


from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
