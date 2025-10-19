"""
Device Manager - Automatic CPU/GPU Detection and Management
Handles device allocation with graceful CPU fallback
"""

import torch
import warnings

class DeviceManager:
    """
    Manages compute devices with automatic CPU/GPU detection.
    Provides safe fallback to CPU if CUDA unavailable.
    """
    
    def __init__(self, prefer_gpu=True):
        """
        Initialize device manager.
        
        Args:
            prefer_gpu: Attempt to use GPU if available
        """
        self.prefer_gpu = prefer_gpu
        self.device = self._get_device()
        self.device_type = "cuda" if self.device.type == "cuda" else "cpu"
        
        # Print device info
        self._print_device_info()
    
    def _get_device(self):
        """Detect and return best available device."""
        if self.prefer_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            # Set memory allocation strategy
            torch.cuda.empty_cache()
            return device
        else:
            if self.prefer_gpu:
                warnings.warn("CUDA not available. Falling back to CPU.")
            return torch.device("cpu")
    
    def _print_device_info(self):
        """Print device information."""
        print("=" * 60)
        print("🖥️  Device Information")
        print("=" * 60)
        print(f"Device Type: {self.device_type.upper()}")
        
        if self.device_type == "cuda":
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("Running on CPU (may be slower for large models)")
        
        print(f"PyTorch Version: {torch.__version__}")
        print("=" * 60 + "\n")
    
    def get_device(self):
        """Return current device."""
        return self.device
    
    def to_device(self, model):
        """
        Move model to current device.
        
        Args:
            model: PyTorch model or tensor
            
        Returns:
            Model on device
        """
        return model.to(self.device)
    
    def empty_cache(self):
        """Clear GPU cache if using CUDA."""
        if self.device_type == "cuda":
            torch.cuda.empty_cache()
    
    def get_memory_info(self):
        """
        Get current memory usage.
        
        Returns:
            Dictionary with memory info
        """
        if self.device_type == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "device": "cuda"
            }
        else:
            return {"device": "cpu", "note": "CPU memory managed by system"}
    
    @property
    def is_cuda(self):
        """Check if using CUDA."""
        return self.device_type == "cuda"
    
    @property
    def is_cpu(self):
        """Check if using CPU."""
        return self.device_type == "cpu"


# Global device manager instance
_device_manager = None

def get_device_manager(prefer_gpu=True):
    """
    Get or create global device manager instance.
    
    Args:
        prefer_gpu: Prefer GPU if available
        
    Returns:
        DeviceManager instance
    """
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager(prefer_gpu=prefer_gpu)
    return _device_manager


def get_device():
    """Quick access to current device."""
    return get_device_manager().get_device()