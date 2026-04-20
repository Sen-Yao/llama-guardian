"""
配置模块测试
"""

import os
import tempfile

from src.config import AppConfig, _deep_merge, load_config


def test_deep_merge():
    """测试字典深度合并"""
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}, "e": 5}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": {"c": 99, "d": 3}, "e": 5}


def test_default_config():
    """测试默认配置值"""
    config = AppConfig()
    assert config.server.host == "0.0.0.0"
    assert config.server.port == 8000
    assert config.llama_server.port == 11434
    assert config.vram.size_multiplier == 1.4
    assert config.vram.safety_margin_mb == 2000
    assert config.cleanup.idle_timeout_seconds == 300
    assert config.concurrency.max_concurrent_requests == 0


def test_load_config_from_yaml():
    """测试从 YAML 文件加载配置"""
    yaml_content = """
server:
  host: "127.0.0.1"
  port: 9090
vram:
  size_multiplier: 1.6
cleanup:
  idle_timeout_seconds: 600
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        config = load_config(yaml_path)
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 9090
        assert config.vram.size_multiplier == 1.6
        assert config.cleanup.idle_timeout_seconds == 600
        # 未覆盖的值应保持默认
        assert config.vram.safety_margin_mb == 2000
    finally:
        os.unlink(yaml_path)


def test_load_config_missing_file():
    """测试加载不存在的配置文件（应使用默认值）"""
    config = load_config("/nonexistent/path/config.yaml")
    assert config.server.port == 8000  # 默认值
