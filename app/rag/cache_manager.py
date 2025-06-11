"""
多层缓存管理器
支持内存缓存、磁盘缓存的分层缓存策略
"""
import hashlib
import json
import time
import pickle
import os
from typing import Any, Optional, Callable, Dict
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class CacheLevel:
    """缓存级别"""
    MEMORY = "memory"      # 内存缓存（最快）
    DISK = "disk"          # 磁盘缓存（中等）

class CacheManager:
    """多层缓存管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'memory_size': 1000,
            'disk_cache_dir': 'cache',
            'default_ttl': 3600  # 1小时
        }
        
        # 内存缓存 (LRU)
        self.memory_cache = {}
        self.memory_times = {}
        self.memory_max_size = self.config['memory_size']
        
        # 磁盘缓存目录
        self.disk_cache_dir = self.config['disk_cache_dir']
        os.makedirs(self.disk_cache_dir, exist_ok=True)
        
        logger.info(f"🗄️ 缓存管理器初始化完成，内存缓存大小: {self.memory_max_size}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值"""
        result = self.get_from_memory(key)
        return result if result is not None else default
    
    def set(self, key: str, value: Any, ttl: int = None):
        """设置缓存值"""
        self.set_to_memory(key, value)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息（简化版本）"""
        return self.get_cache_stats()
    
    def _generate_key(self, prefix: str, query: str, **kwargs) -> str:
        """生成缓存键"""
        # 将查询和参数组合生成唯一键
        content = f"{prefix}:{query}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_expired(self, timestamp: float, ttl: int) -> bool:
        """检查是否过期"""
        return time.time() - timestamp > ttl
    
    def get_from_memory(self, key: str, ttl: int = None) -> Optional[Any]:
        """从内存缓存获取"""
        if key not in self.memory_cache:
            return None
        
        # 检查是否过期
        if ttl and key in self.memory_times:
            if self._is_expired(self.memory_times[key], ttl):
                del self.memory_cache[key]
                del self.memory_times[key]
                return None
        
        return self.memory_cache[key]
    
    def set_to_memory(self, key: str, value: Any):
        """设置内存缓存"""
        # 如果超过最大大小，移除最老的项
        if len(self.memory_cache) >= self.memory_max_size:
            oldest_key = min(self.memory_times.keys(), key=lambda k: self.memory_times[k])
            del self.memory_cache[oldest_key]
            del self.memory_times[oldest_key]
        
        self.memory_cache[key] = value
        self.memory_times[key] = time.time()
    
    def get_from_disk(self, key: str, ttl: int = None) -> Optional[Any]:
        """从磁盘缓存获取"""
        cache_file = os.path.join(self.disk_cache_dir, f"{key}.pkl")
        
        if not os.path.exists(cache_file):
            return None
        
        # 检查文件是否过期
        if ttl:
            file_time = os.path.getmtime(cache_file)
            if self._is_expired(file_time, ttl):
                os.remove(cache_file)
                return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"⚠️ 磁盘缓存读取失败: {e}")
            return None
    
    def set_to_disk(self, key: str, value: Any):
        """设置磁盘缓存"""
        cache_file = os.path.join(self.disk_cache_dir, f"{key}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"⚠️ 磁盘缓存写入失败: {e}")
    
    def get_or_compute(self, 
                      prefix: str,
                      query: str, 
                      compute_func: Callable,
                      ttl: int = None,
                      use_disk: bool = True,
                      **kwargs) -> Any:
        """获取缓存或计算结果"""
        # 生成缓存键
        cache_key = self._generate_key(prefix, query, **kwargs)
        ttl = ttl or self.config['default_ttl']
        
        # 1. 尝试内存缓存
        result = self.get_from_memory(cache_key, ttl)
        if result is not None:
            logger.debug(f"🚀 内存缓存命中: {prefix}")
            return result
        
        # 2. 尝试磁盘缓存
        if use_disk:
            result = self.get_from_disk(cache_key, ttl)
            if result is not None:
                logger.debug(f"💾 磁盘缓存命中: {prefix}")
                # 同时更新内存缓存
                self.set_to_memory(cache_key, result)
                return result
        
        # 3. 计算新结果
        logger.debug(f"🔄 缓存未命中，开始计算: {prefix}")
        start_time = time.time()
        result = compute_func(query, **kwargs)
        compute_time = time.time() - start_time
        
        # 4. 更新缓存
        self.set_to_memory(cache_key, result)
        if use_disk:
            self.set_to_disk(cache_key, result)
        
        logger.debug(f"✅ 计算完成并缓存: {prefix}, 耗时: {compute_time:.3f}s")
        return result
    
    def clear_cache(self, prefix: str = None):
        """清理缓存"""
        if prefix:
            # 清理特定前缀的缓存
            keys_to_remove = [k for k in self.memory_cache.keys() if k.startswith(prefix)]
            for key in keys_to_remove:
                del self.memory_cache[key]
                if key in self.memory_times:
                    del self.memory_times[key]
            logger.info(f"🧹 已清理前缀为 {prefix} 的缓存")
        else:
            # 清理所有缓存
            self.memory_cache.clear()
            self.memory_times.clear()
            logger.info("🧹 已清理所有内存缓存")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'memory_cache_size': len(self.memory_cache),
            'memory_max_size': self.memory_max_size,
            'disk_cache_dir': self.disk_cache_dir,
            'config': self.config
        }

# 全局缓存管理器实例
cache_manager = CacheManager()
