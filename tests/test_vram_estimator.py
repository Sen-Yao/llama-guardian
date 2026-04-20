"""
显存估算模块测试
"""

import os
import tempfile

import pytest

from src.config import AppConfig, ModelConfig, VramConfig
from src.vram_estimator import VramEstimator


@pytest.fixture
def config_with_model():
    """创建包含测试模型路径的配置"""
    # 创建临时模型文件
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        # 写入 1000 字节（模拟模型文件）
        f.write(b"x" * (100 * 1024 * 1024))  # 100 MB
        model_path = f.name

    config = AppConfig(
        models=[
            ModelConfig(name="test-model", path=model_path),
        ],
        vram=VramConfig(
            size_multiplier=1.4,
            safety_margin_mb=2000,
        ),
    )

    yield config

    # 清理
    os.unlink(model_path)


def test_estimate_required_vram(config_with_model):
    """测试显存估算"""
    estimator = VramEstimator(config_with_model)
    required = estimator.estimate_required_vram("test-model")

    # 100MB * 1.4 + 2000MB = 2140MB
    assert required == 2140


def test_estimate_unknown_model(config_with_model):
    """测试未知模型的显存估算"""
    estimator = VramEstimator(config_with_model)
    required = estimator.estimate_required_vram("nonexistent-model")
    assert required == -1


def test_get_model_path(config_with_model):
    """测试获取模型路径"""
    estimator = VramEstimator(config_with_model)
    path = estimator.get_model_path("test-model")
    assert path is not None
    assert path.endswith(".gguf")


def test_list_models(config_with_model):
    """测试列出模型"""
    estimator = VramEstimator(config_with_model)
    models = estimator.list_models()
    assert len(models) == 1
    assert models[0]["name"] == "test-model"
    assert models[0]["file_size_mb"] == 100
    assert models[0]["estimated_vram_mb"] == 2140
