# -*- coding: utf-8 -*-
"""
Qwen2.5-1.5B-Instruct模型的LangChain包装器 (使用ModelScope)
提供基于ModelScope的Qwen2.5-1.5B-Instruct模型与LangChain框架的集成
"""

from modelscope import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional, List, Any
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation
from pydantic import Field
import logging
import os
import shutil
from config import config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 模块级别的模型缓存
_MODEL_CACHE = {
    'model': None,
    'tokenizer': None,
    'initialized': False
}


class QwenLLM(BaseLLM):
    """
    基于Qwen2.5-1.5B-Instruct的本地LLM实现 (使用ModelScope)
    使用模块级别的模型缓存避免重复初始化
    """
    
    model_name: str = Field(default_factory=lambda: config.qwen_model_name, description="模型名称")
    max_length: int = Field(default_factory=lambda: config.qwen_max_length, description="最大生成长度")
    temperature: float = Field(default_factory=lambda: config.qwen_temperature, description="生成温度")
    device: Any = Field(default=None, description="计算设备")
    
    def __init__(self, **kwargs):
        """初始化Qwen LLM
        
        Args:
            **kwargs: 其他参数
        """
        # 正确调用父类初始化
        super().__init__(**kwargs)
        
        # 设置设备（使用配置系统）
        self.device = torch.device(config.device)
        
        # 如果模型还未初始化，则初始化缓存模型
        if not _MODEL_CACHE['initialized']:
            self._initialize_model()
    
    @property
    def model(self):
        """获取缓存的模型实例"""
        return _MODEL_CACHE['model']
    
    @property
    def tokenizer(self):
        """获取缓存的tokenizer实例"""
        return _MODEL_CACHE['tokenizer']
    
    @classmethod
    def clear_instance(cls):
        """清理缓存模型实例，释放模型资源"""
        try:
            if _MODEL_CACHE['model'] is not None:
                del _MODEL_CACHE['model']
                _MODEL_CACHE['model'] = None
                
            if _MODEL_CACHE['tokenizer'] is not None:
                del _MODEL_CACHE['tokenizer']
                _MODEL_CACHE['tokenizer'] = None
                
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            _MODEL_CACHE['initialized'] = False
            logger.info("🧹 已清理Qwen缓存模型实例和CUDA缓存")
        except Exception as e:
            logger.warning(f"清理模型实例时出现警告: {e}")
    
    def _initialize_model(self):
        """初始化缓存的Qwen模型 (使用ModelScope)"""
        if _MODEL_CACHE['initialized']:
            logger.info("🔄 使用已存在的Qwen缓存模型实例")
            return
            
        import time
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"正在从ModelScope加载Qwen模型: qwen/Qwen2.5-1.5B-Instruct (尝试 {attempt + 1}/{max_retries})")
                
                # 清理CUDA缓存以释放显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("🧹 已清理CUDA缓存")
                
                # 加载tokenizer
                _MODEL_CACHE['tokenizer'] = AutoTokenizer.from_pretrained(
                    "qwen/Qwen2.5-1.5B-Instruct",
                    trust_remote_code=True,
                    revision='master',
                    local_files_only=False,
                    force_download=False
                )
                
                # 使用优化配置加载模型以节省显存
                _MODEL_CACHE['model'] = AutoModelForCausalLM.from_pretrained(
                    "qwen/Qwen2.5-1.5B-Instruct",
                    torch_dtype=torch.float16,  # 使用半精度
                    low_cpu_mem_usage=True,     # 降低CPU内存使用
                    trust_remote_code=True,
                    revision='master',
                    local_files_only=False,
                    force_download=False
                )
                
                # 手动移动模型到设备
                _MODEL_CACHE['model'] = _MODEL_CACHE['model'].to(self.device)
                
                # 设置为评估模式
                _MODEL_CACHE['model'].eval()
                _MODEL_CACHE['initialized'] = True
                
                logger.info("✅ Qwen缓存模型从ModelScope加载成功")
                return
                
            except PermissionError as e:
                if "另一个程序正在使用此文件" in str(e) or "WinError 32" in str(e):
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ 文件被占用，等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                        continue
                    else:
                        logger.error("❌ 多次重试后仍然无法访问模型文件，请检查是否有其他程序在使用模型")
                        raise e
                else:
                    logger.error(f"❌ 权限错误: {str(e)}")
                    raise e
            except Exception as e:
                error_msg = str(e)
                # 检查是否是文件损坏或校验失败
                if ("integrity check failed" in error_msg or 
                    "control character" in error_msg or
                    "download may be incomplete" in error_msg):
                    logger.warning(f"⚠️ 检测到文件损坏，清理缓存后重试: {error_msg}")
                    # 清理ModelScope缓存
                    self._clear_modelscope_cache()
                
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ 加载失败，等待 {retry_delay} 秒后重试: {error_msg}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    logger.error(f"❌ Qwen模型从ModelScope加载失败: {error_msg}")
                    raise e
    
    def _clear_modelscope_cache(self):
        """清理ModelScope缓存中的损坏文件"""
        try:
            import shutil
            import os
            
            # ModelScope缓存路径
            cache_dir = os.path.expanduser("~/.cache/modelscope/hub/models/qwen/Qwen2.5-1.5B-Instruct")
            temp_dir = os.path.expanduser("~/.cache/modelscope/hub/models/._____temp/qwen/Qwen2.5-1.5B-Instruct")
            
            # 清理临时目录
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"已清理临时缓存目录: {temp_dir}")
            
            # 清理主缓存目录中的损坏文件
            if os.path.exists(cache_dir):
                for file_name in ["tokenizer_config.json", "tokenizer.json"]:
                    file_path = os.path.join(cache_dir, file_name)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"已删除损坏文件: {file_path}")
                        
        except Exception as e:
            logger.warning(f"清理缓存时出错: {str(e)}")
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型标识"""
        return "qwen"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        调用Qwen模型生成文本
        
        Args:
            prompt: 输入提示文本
            stop: 停止词列表
            run_manager: 回调管理器
            **kwargs: 其他生成参数
            
        Returns:
            生成的文本响应
        """
        try:
            if self.model is None or self.tokenizer is None:
                logger.error("模型或tokenizer未正确初始化")
                return "模型未正确初始化"
            
            logger.info(f"收到提示: {prompt[:100]}...")
            
            # 构建消息格式
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # 应用聊天模板
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            logger.info(f"格式化后的输入: {text[:200]}...")
            
            # 编码输入
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            # 生成响应
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=kwargs.get("max_length", self.max_length),
                    temperature=kwargs.get("temperature", self.temperature),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码响应
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            logger.info(f"生成的响应: {response}")
            
            # 检查响应是否为空
            if not response or response.strip() == "":
                logger.warning("模型生成了空响应")
                return "根据提供的文档，我无法找到相关信息。"
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Qwen生成失败: {str(e)}")
            return f"生成失败: {str(e)}"
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """
        生成多个提示的响应
        
        Args:
            prompts: 提示文本列表
            stop: 停止词列表
            run_manager: 回调管理器
            **kwargs: 其他生成参数
            
        Returns:
            LLM结果对象
        """
        generations = []
        for prompt in prompts:
            try:
                text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
                generations.append([Generation(text=text)])
            except Exception as e:
                logger.error(f"生成失败: {str(e)}")
                generations.append([Generation(text=f"生成失败: {str(e)}")])
        
        return LLMResult(generations=generations)
    
    def chat(
        self,
        message: str,
        history: List[tuple] = None,
        **kwargs
    ) -> str:
        """
        进行对话
        
        Args:
            message: 用户消息
            history: 对话历史
            **kwargs: 其他参数
            
        Returns:
            模型回复
        """
        if history is None:
            history = []
            
        try:
            # 构建消息历史
            messages = []
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
            
            # 添加当前用户消息
            messages.append({"role": "user", "content": message})
            
            # 应用聊天模板
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 编码输入
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            # 生成响应
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=kwargs.get("max_length", self.max_length),
                    temperature=kwargs.get("temperature", self.temperature),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码响应
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return response
            
        except Exception as e:
            logger.error(f"对话生成失败: {str(e)}")
            return f"对话生成失败: {str(e)}"