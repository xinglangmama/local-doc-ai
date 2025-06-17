"""配置管理模块

用于管理应用程序的配置选项，包括设备选择、模型配置等。
"""

import torch
import logging
import yaml
import os
from typing import Dict, Any, Optional


class Config:
    """配置管理类"""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config_data = self._load_config()
        
        # 初始化日志
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            else:
                logging.warning(f"配置文件 {self.config_file} 不存在，使用默认配置")
                return self._get_default_config()
        except Exception as e:
            logging.error(f"加载配置文件失败: {e}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "device": {"mode": "auto"},
            "model": {
                "qwen": {
                    "name": "Qwen/Qwen2.5-7B-Instruct",
                    "max_length": 8192,
                    "temperature": 0.7
                },
                "embedding": {
                    "name": "BAAI/bge-large-zh-v1.5",
                    "cache_folder": "./models",
                    "trust_remote_code": True
                }
            },
            "vector_db": {
                "type": "faiss",
                "index_type": "IndexFlatIP",
                "dimension": 1024
            },
            "chat": {
                "max_history": 10,
                "context_window": 4000
            },
            "document": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "supported_formats": [".txt", ".pdf", ".docx", ".md"]
            },
            "app": {
                "title": "本地文档AI助手",
                "page_icon": "🤖",
                "layout": "wide"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "cache": {
                "model_cache_size": 2,
                "clear_on_exit": False
            }
        }
    
    def save_config(self):
        """保存配置到YAML文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False)
            logging.info(f"配置已保存到 {self.config_file}")
        except Exception as e:
            logging.error(f"保存配置文件失败: {e}")
    
    # 属性访问器
    @property
    def device_mode(self) -> str:
        return self.config_data.get("device", {}).get("mode", "auto")
    
    @property
    def qwen_model_name(self) -> str:
        return self.config_data.get("model", {}).get("qwen", {}).get("name", "Qwen/Qwen2.5-7B-Instruct")
    
    @property
    def qwen_max_length(self) -> int:
        return self.config_data.get("model", {}).get("qwen", {}).get("max_length", 8192)
    
    @property
    def qwen_temperature(self) -> float:
        return self.config_data.get("model", {}).get("qwen", {}).get("temperature", 0.7)
    
    @property
    def embedding_model_name(self) -> str:
        return self.config_data.get("model", {}).get("embedding", {}).get("name", "BAAI/bge-large-zh-v1.5")
    
    @property
    def embedding_cache_folder(self) -> str:
        return self.config_data.get("model", {}).get("embedding", {}).get("cache_folder", "./models")
    
    @property
    def vector_db_type(self) -> str:
        return self.config_data.get("vector_db", {}).get("type", "faiss")
    
    @property
    def vector_dimension(self) -> int:
        return self.config_data.get("vector_db", {}).get("dimension", 1024)
    
    @property
    def max_chat_history(self) -> int:
        return self.config_data.get("chat", {}).get("max_history", 10)
    
    @property
    def context_window(self) -> int:
        return self.config_data.get("chat", {}).get("context_window", 4000)
    
    def _setup_logging(self):
        """设置日志配置"""
        log_config = self.config_data.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO").upper())
        format_str = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[
                logging.StreamHandler()
            ]
        )
    
    def _get_device(self) -> str:
        """根据配置和硬件情况自动选择设备"""
        mode = self.device_mode
        if mode == "cpu":
            return "cpu"
        elif mode == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            else:
                logging.warning("请求使用CUDA但CUDA不可用，回退到CPU")
                return "cpu"
        else:  # auto模式
            return "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def device(self) -> str:
        """获取当前设备"""
        return self._get_device()
    
    def set_device_mode(self, mode: str):
        """设置设备模式"""
        if mode not in ["auto", "cuda", "cpu"]:
            raise ValueError(f"无效的设备模式: {mode}")
        
        # 更新配置数据
        if "device" not in self.config_data:
            self.config_data["device"] = {}
        self.config_data["device"]["mode"] = mode
        
        # 保存配置
        self.save_config()
        
        current_device = self._get_device()
        logging.info(f"设备模式已设置为: {mode}, 当前使用设备: {current_device}")
    
    def get_device_info(self) -> dict:
        """获取设备信息"""
        current_device = self._get_device()
        info = {
            "device_mode": self.device_mode,
            "current_device": current_device,
            "cuda_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        
        return info
    
    def update_config(self, section: str, key: str, value: Any):
        """更新配置项"""
        if section not in self.config_data:
            self.config_data[section] = {}
        self.config_data[section][key] = value
        self.save_config()
    
    def get_config(self, section: str, key: str = None, default=None):
        """获取配置项"""
        section_data = self.config_data.get(section, {})
        if key is None:
            return section_data
        return section_data.get(key, default)


# 创建全局配置实例
config = Config()