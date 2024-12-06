import os
import psutil
import gc
import numpy as np
from typing import Dict, Optional
import logging
from dataclasses import dataclass
import torch

@dataclass
class MemoryStats:
    total: float  # Total memory in GB
    available: float  # Available memory in GB
    used: float  # Used memory in GB
    percent: float  # Percentage used

class MemoryManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process(os.getpid())
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        vm = psutil.virtual_memory()
        return MemoryStats(
            total=vm.total / (1024**3),  # Convert to GB
            available=vm.available / (1024**3),
            used=vm.used / (1024**3),
            percent=vm.percent
        )
    
    def check_memory_usage(self, threshold: float = 90.0) -> bool:
        """Check if memory usage is below threshold"""
        stats = self.get_memory_stats()
        if stats.percent > threshold:
            self.logger.warning(
                f"Memory usage ({stats.percent:.1f}%) above threshold ({threshold}%)"
            )
            return False
        return True
    
    def optimize_chunk_size(
        self,
        initial_size: int,
        item_size: float,
        target_memory_gb: float
    ) -> int:
        """Calculate optimal chunk size based on memory constraints"""
        stats = self.get_memory_stats()
        available_memory = min(stats.available, target_memory_gb)
        max_items = int(available_memory * 1024**3 / item_size)
        return min(initial_size, max_items)
    
    def clear_memory(self, force_gpu: bool = True):
        """Clear unused memory"""
        gc.collect()
        if force_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def memory_efficient_operation(self, func, *args, **kwargs):
        """Decorator for memory-efficient operations"""
        def wrapper(*args, **kwargs):
            initial_memory = self.process.memory_info().rss
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.clear_memory()
                
                final_memory = self.process.memory_info().rss
                memory_diff = (final_memory - initial_memory) / (1024**2)
                self.logger.debug(f"Memory change: {memory_diff:.2f}MB")
        
        return wrapper
    
    @staticmethod
    def estimate_tensor_size(shape: tuple, dtype: torch.dtype) -> float:
        """Estimate memory size of tensor in bytes"""
        elem_size = {
            torch.float32: 4,
            torch.float64: 8,
            torch.int32: 4,
            torch.int64: 8
        }.get(dtype, 4)
        
        return np.prod(shape) * elem_size
    
    def optimize_batch_size(
        self,
        model: torch.nn.Module,
        sample_input: Dict[str, torch.Tensor],
        target_memory_gb: float,
        min_batch: int = 32,
        max_batch: int = 512
    ) -> int:
        """Find optimal batch size through binary search"""
        self.clear_memory()
        
        def test_batch_size(batch_size: int) -> bool:
            try:
                # Scale sample input to batch size
                batch = {
                    k: v.repeat(batch_size, *[1]*(v.dim()-1))
                    for k, v in sample_input.items()
                }
                
                # Test forward and backward pass
                with torch.no_grad():
                    output = model(batch)
                    if isinstance(output, dict):
                        loss = sum(v.sum() for v in output.values() 
                                 if isinstance(v, torch.Tensor))
                    else:
                        loss = output.sum()
                
                stats = self.get_memory_stats()
                return stats.available > target_memory_gb
                
            except RuntimeError:  # Out of memory
                return False
            finally:
                self.clear_memory()
        
        # Binary search for largest viable batch size
        left, right = min_batch, max_batch
        optimal_batch = min_batch
        
        while left <= right:
            mid = (left + right) // 2
            if test_batch_size(mid):
                optimal_batch = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return optimal_batch
    
    def get_memory_profiler(self):
        """Get memory profiler context manager"""
        from contextlib import contextmanager
        import time
        
        @contextmanager
        def profile_memory():
            start_mem = self.process.memory_info().rss
            start_time = time.time()
            yield
            end_mem = self.process.memory_info().rss
            end_time = time.time()
            
            mem_diff = (end_mem - start_mem) / (1024**2)
            time_diff = end_time - start_time
            
            self.logger.info(
                f"Memory change: {mem_diff:.2f}MB in {time_diff:.2f}s"
            )
        
        return profile_memory