"""
显存估算模块
根据模型文件大小自动推算运行所需显存
"""

import logging
from pathlib import Path

from .config import AppConfig, ModelConfig

logger = logging.getLogger("llama-guardian")


class VramEstimator:
    """基于模型文件大小的显存估算器"""

    def __init__(self, config: AppConfig):
        self.config = config
        self._model_cache: dict[str, int] = {}  # model_name -> file_size_bytes

    def _get_model_config(self, model_name: str) -> ModelConfig | None:
        """根据模型名称查找模型配置"""
        for m in self.config.models:
            if m.name == model_name:
                return m
        return None

    def _get_model_file_size_mb(self, model_path: str) -> int:
        """
        获取模型文件大小（MB）

        Args:
            model_path: 模型文件路径

        Returns:
            文件大小（MB），向上取整
        """
        path = Path(model_path)
        if not path.exists():
            logger.warning(f"模型文件不存在: {model_path}")
            return 0
        size_bytes = path.stat().st_size
        size_mb = (size_bytes + 1024 * 1024 - 1) // (1024 * 1024)  # 向上取整
        return size_mb

    def estimate_required_vram(self, model_name: str) -> int:
        """
        估算运行指定模型所需的显存（MB）

        计算公式：模型文件大小 × size_multiplier + safety_margin_mb

        Args:
            model_name: 模型名称（对应 config 中的 name 字段）

        Returns:
            预估所需显存（MB），如果模型未找到返回 -1
        """
        model_config = self._get_model_config(model_name)
        if model_config is None:
            logger.error(f"未找到模型配置: {model_name}")
            return -1

        file_size_mb = self._get_model_file_size_mb(model_config.path)
        if file_size_mb == 0:
            logger.error(f"无法获取模型文件大小: {model_config.path}")
            return -1

        required = int(file_size_mb * self.config.vram.size_multiplier) + self.config.vram.safety_margin_mb

        logger.info(
            f"显存估算: 模型={model_name}, "
            f"文件大小={file_size_mb}MB, "
            f"倍率={self.config.vram.size_multiplier}, "
            f"安全边际={self.config.vram.safety_margin_mb}MB, "
            f"预估需要={required}MB"
        )

        return required

    def get_model_path(self, model_name: str) -> str | None:
        """获取模型文件路径"""
        model_config = self._get_model_config(model_name)
        if model_config:
            return model_config.path
        return None

    def list_models(self) -> list[dict]:
        """
        列出所有已配置的模型及其估算显存

        Returns:
            模型信息列表
        """
        result = []
        for model in self.config.models:
            file_size_mb = self._get_model_file_size_mb(model.path)
            required_vram = int(file_size_mb * self.config.vram.size_multiplier) + self.config.vram.safety_margin_mb if file_size_mb > 0 else -1
            result.append({
                "name": model.name,
                "path": model.path,
                "file_size_mb": file_size_mb,
                "estimated_vram_mb": required_vram,
            })
        return result