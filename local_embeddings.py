# -*- coding: utf-8 -*-
"""  
本地嵌入模型包装器
用于将 SentenceTransformer 模型包装为 LangChain 兼容的嵌入模型
支持 CUDA GPU 加速计算
"""

import logging
import torch
import numpy as np
import os
import pickle
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

# ModelScope支持
try:
    from modelscope import snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    print("警告: ModelScope未安装，将使用默认的sentence-transformers加载方式")

# 配置日志 - 避免重复配置
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# 全局模型缓存
_MODEL_CACHE = {}

class LocalEmbeddings(Embeddings):
    """
    本地嵌入模型类，支持 CUDA GPU 加速和ModelScope模型
    
    Features:
    - 全局模型缓存，避免重复加载
    - 自动设备检测，优先使用GPU
    - CUDA 内存管理和优化
    - 批量处理支持
    - 支持ModelScope模型自动下载
    - 支持gte-Qwen2-1.5B-instruct等高性能中文向量模型
    
    Optimizations:
    - ModelScope优先，自动回退到sentence-transformers
    - 直接GPU加载，智能设备选择
    - 保留核心CUDA内存优化
    - 支持trust_remote_code参数
    """
    
    def __init__(self, 
                 model_name: str = 'iic/gte_Qwen2-1.5B-instruct',
                 device: Optional[str] = None,
                 batch_size: int = 32,
                 normalize_embeddings: bool = True,
                 show_progress_bar: bool = False):
        """
        初始化本地嵌入模型
        
        Args:
            model_name: 模型名称或路径
            device: 计算设备 ('cuda:0', 'cuda:1', 'cpu' 等)，None 为自动检测
            batch_size: 批处理大小，用于大量文本处理
            normalize_embeddings: 是否标准化嵌入向量
            show_progress_bar: 是否显示进度条
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        
        # 设备检测和设置
        self.device = self._get_optimal_device(device)
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        self._initialize_model()
    
    def _get_optimal_device(self, preferred_device: Optional[str] = None) -> str:
        """
        获取最优计算设备（优化版本）
        
        Args:
            preferred_device: 首选设备
            
        Returns:
            str: 最终使用的设备名称
        """
        # 如果指定了设备，优先使用
        if preferred_device:
            if preferred_device.startswith('cuda') and torch.cuda.is_available():
                try:
                    device_id = int(preferred_device.split(':')[1]) if ':' in preferred_device else 0
                    if device_id < torch.cuda.device_count():
                        return preferred_device
                    else:
                        return 'cuda:0'
                except (ValueError, IndexError):
                    pass
            elif preferred_device == 'cpu':
                return 'cpu'
        
        # 自动检测最优设备
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            
            # 快速选择显存最多的设备
            best_device = 0
            max_memory = 0
            for i in range(device_count):
                memory = torch.cuda.get_device_properties(i).total_memory
                if memory > max_memory:
                    max_memory = memory
                    best_device = i
            
            selected_device = f'cuda:{best_device}'
            # 只在首次检测时输出详细信息
            if not hasattr(self, '_device_logged'):
                device_name = torch.cuda.get_device_name(best_device)
                logger.info(f"选择设备: {selected_device} ({device_name}, {max_memory / 1024**3:.1f} GB)")
                self._device_logged = True
            return selected_device
        else:
            if not hasattr(self, '_device_logged'):
                logger.info("使用 CPU 设备")
                self._device_logged = True
            return 'cpu'
    
    def _initialize_model(self):
        """
        初始化 SentenceTransformer 模型（支持ModelScope）
        """
        # 创建缓存键
        cache_key = f"{self.model_name}_{self.device}"
        
        # 检查全局缓存
        if cache_key in _MODEL_CACHE:
            logger.info(f"从缓存加载模型: {self.model_name}")
            self.model = _MODEL_CACHE[cache_key]
            return
        
        logger.info(f"加载模型到 {self.device}: {self.model_name}")
        
        # 尝试从ModelScope下载模型
        model_path = self._get_model_path()
        
        # 加载模型到指定设备
        self.model = SentenceTransformer(model_path, device=self.device, trust_remote_code=True)
        self.model.eval()
        
        # 如果使用 CUDA，进行内存优化
        if self.device.startswith('cuda'):
            self._optimize_cuda_memory()
        
        # 缓存模型
        _MODEL_CACHE[cache_key] = self.model
        
        logger.info(f"✅ 模型加载成功")
    
    def _get_model_path(self):
        """
        获取模型路径，优先从ModelScope下载
        
        Returns:
            str: 模型路径或模型名称
        """
        # 如果是ModelScope格式的模型名称且ModelScope可用
        if MODELSCOPE_AVAILABLE and '/' in self.model_name and not self.model_name.startswith('sentence-transformers/'):
            try:
                logger.info(f"正在从ModelScope下载模型: {self.model_name}")
                model_dir = snapshot_download(self.model_name)
                logger.info(f"ModelScope模型下载完成: {model_dir}")
                return model_dir
            except Exception as e:
                logger.warning(f"ModelScope下载失败: {str(e)}，将使用原始模型名称")
                return self.model_name
        else:
            # 直接使用模型名称（适用于sentence-transformers格式）
            return self.model_name
    
    def _optimize_cuda_memory(self):
        """
        优化 CUDA 内存使用
        """
        if torch.cuda.is_available() and self.device.startswith('cuda'):
            # 清理缓存
            torch.cuda.empty_cache()
            
            # 设置内存分配策略
            torch.cuda.set_per_process_memory_fraction(0.8)  # 使用 80% 显存
            
            # 记录内存信息
            device_id = int(self.device.split(':')[1]) if ':' in self.device else 0
            memory_allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device_id) / 1024**3
            
            logger.info(f"CUDA 内存 - 已分配: {memory_allocated:.2f} GB, 已保留: {memory_reserved:.2f} GB")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档（优化版本）
        
        Args:
            texts: 文档文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not texts:
            return []
        
        try:
            # 只在处理大量文档时显示日志
            if len(texts) > 10:
                logger.info(f"处理 {len(texts)} 个文档")
            
            # 批量编码以提高效率
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar and len(texts) > 50,  # 只在大量文档时显示进度条
                normalize_embeddings=self.normalize_embeddings,
                convert_to_tensor=False  # 直接返回 numpy 数组
            )
            
            # 转换为列表格式
            result = embeddings.tolist()
            
            # 简化的性能统计
            if len(texts) > 10:
                logger.info(f"✅ 文档嵌入完成")
            
            # CUDA 内存清理（减少频率）
            if self.device.startswith('cuda') and len(texts) > 20:
                torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"文档嵌入失败: {str(e)}")
            # 尝试单个处理
            return self._fallback_embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询文本（优化版本）
        
        Args:
            text: 查询文本
            
        Returns:
            List[float]: 嵌入向量
        """
        try:
            # 编码单个文本（移除性能统计以提高速度）
            embedding = self.model.encode(
                [text],
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_tensor=False
            )
            
            return embedding[0].tolist()
            
        except Exception as e:
            logger.error(f"查询嵌入失败: {str(e)}")
            raise RuntimeError(f"无法处理查询: {str(e)}")
    
    def _fallback_embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        回退方案：逐个处理文档
        
        Args:
            texts: 文档文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        logger.warning("使用回退方案逐个处理文档")
        results = []
        
        for i, text in enumerate(texts):
            try:
                embedding = self.embed_query(text)
                results.append(embedding)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"已处理 {i + 1}/{len(texts)} 个文档")
                    
            except Exception as e:
                logger.error(f"处理第 {i+1} 个文档失败: {str(e)}")
                # 使用零向量作为占位符
                if results:
                    zero_embedding = [0.0] * len(results[0])
                else:
                    zero_embedding = [0.0] * 384  # 默认维度
                results.append(zero_embedding)
        
        return results