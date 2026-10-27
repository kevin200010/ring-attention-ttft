"""Ring Attention TTFT experiments for long-context RAG on Apple Silicon."""

from .attention import LAYOUTS, device_work, reference_attention, ring_attention, segments

__all__ = ["LAYOUTS", "device_work", "reference_attention", "ring_attention", "segments"]
__version__ = "0.2.0"
