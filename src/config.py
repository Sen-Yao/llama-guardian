"""
配置加载模块
优先级：命令行参数 > 环境变量 > config.yaml > 默认值
"""

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class LlamaServerConfig:
    binary_path: str = "./llama-server"
    host: str = "127.0.0.1"
    port: int = 11434
    default_model_path: Optional[str] = None
    context_size: int = 4096
    extra_args: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    name: str = ""
    path: str = ""


@dataclass
class VramConfig:
    size_multiplier: float = 1.4
    safety_margin_mb: int = 2000


@dataclass
class CleanupConfig:
    idle_timeout_seconds: int = 300
    check_interval_seconds: int = 10


@dataclass
class ConcurrencyConfig:
    max_concurrent_requests: int = 0


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"


@dataclass
class AppConfig:
    llama_server: LlamaServerConfig = field(default_factory=LlamaServerConfig)
    models: list[ModelConfig] = field(default_factory=list)
    vram: VramConfig = field(default_factory=VramConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并两个字典，override 覆盖 base"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dataclass_to_dict(obj) -> dict:
    """将嵌套 dataclass 转为字典"""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(v) for k, v in obj.__dict__.items()}
    return obj


def _dict_to_dataclass(cls, data: dict):
    """将字典转为嵌套 dataclass"""
    if not isinstance(data, dict):
        return data
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key in field_types:
            ftype = field_types[key]
            # 处理 Optional 类型
            origin = getattr(ftype, "__origin__", None)
            if origin is list and isinstance(value, list):
                kwargs[key] = value
            elif hasattr(ftype, "__dataclass_fields__") and isinstance(value, dict):
                kwargs[key] = _dict_to_dataclass(ftype, value)
            else:
                kwargs[key] = value
    return cls(**kwargs)


def _load_yaml(path: str) -> dict:
    """加载 YAML 配置文件"""
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data else {}


def _load_env_overrides() -> dict:
    """从环境变量加载配置覆盖
    格式: LLAMA_SERVER__BINARY_PATH=/usr/bin/llama-server
    双下划线分隔层级
    """
    overrides = {}

    # 环境变量到配置路径的映射
    env_mapping = {
        # llama_server
        "LLAMA_SERVER__BINARY_PATH": ("llama_server", "binary_path"),
        "LLAMA_SERVER__HOST": ("llama_server", "host"),
        "LLAMA_SERVER__PORT": ("llama_server", "port"),
        "LLAMA_SERVER__DEFAULT_MODEL_PATH": ("llama_server", "default_model_path"),
        "LLAMA_SERVER__CONTEXT_SIZE": ("llama_server", "context_size"),
        # vram
        "VRAM__SIZE_MULTIPLIER": ("vram", "size_multiplier"),
        "VRAM__SAFETY_MARGIN_MB": ("vram", "safety_margin_mb"),
        # cleanup
        "CLEANUP__IDLE_TIMEOUT_SECONDS": ("cleanup", "idle_timeout_seconds"),
        "CLEANUP__CHECK_INTERVAL_SECONDS": ("cleanup", "check_interval_seconds"),
        # concurrency
        "CONCURRENCY__MAX_CONCURRENT_REQUESTS": ("concurrency", "max_concurrent_requests"),
        # server
        "SERVER__HOST": ("server", "host"),
        "SERVER__PORT": ("server", "port"),
        # logging
        "LOGGING__LEVEL": ("logging", "level"),
        "LOGGING__FORMAT": ("logging", "format"),
    }

    for env_key, path_tuple in env_mapping.items():
        value = os.environ.get(env_key)
        if value is not None:
            # 类型转换
            section, key = path_tuple
            if section not in overrides:
                overrides[section] = {}

            # 根据默认值类型进行转换
            defaults = AppConfig()
            default_section = getattr(defaults, section)
            default_value = getattr(default_section, key)

            if isinstance(default_value, bool):
                overrides[section][key] = value.lower() in ("true", "1", "yes")
            elif isinstance(default_value, int):
                overrides[section][key] = int(value)
            elif isinstance(default_value, float):
                overrides[section][key] = float(value)
            else:
                overrides[section][key] = value

    return overrides


def _parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Llama Guardian - GPU 资源调度中间层")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--llama-server.binary-path", type=str, dest="llama_binary_path")
    parser.add_argument("--llama-server.host", type=str, dest="llama_host")
    parser.add_argument("--llama-server.port", type=int, dest="llama_port")
    parser.add_argument("--server.host", type=str, dest="server_host")
    parser.add_argument("--server.port", type=int, dest="server_port")
    parser.add_argument("--cleanup.idle-timeout-seconds", type=int, dest="idle_timeout")
    parser.add_argument("--logging.level", type=str, dest="log_level")
    return parser.parse_known_args()[0]


def _args_to_overrides(args: argparse.Namespace) -> dict:
    """将命令行参数转为配置覆盖字典"""
    overrides = {}

    if args.llama_binary_path:
        overrides.setdefault("llama_server", {})["binary_path"] = args.llama_binary_path
    if args.llama_host:
        overrides.setdefault("llama_server", {})["host"] = args.llama_host
    if args.llama_port:
        overrides.setdefault("llama_server", {})["port"] = args.llama_port
    if args.server_host:
        overrides.setdefault("server", {})["host"] = args.server_host
    if args.server_port:
        overrides.setdefault("server", {})["port"] = args.server_port
    if args.idle_timeout:
        overrides.setdefault("cleanup", {})["idle_timeout_seconds"] = args.idle_timeout
    if args.log_level:
        overrides.setdefault("logging", {})["level"] = args.log_level

    return overrides


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """
    加载配置，按优先级合并：
    1. 命令行参数（最高）
    2. 环境变量
    3. config.yaml
    4. 默认值（最低）
    """
    # 1. 默认值
    defaults_dict = _dataclass_to_dict(AppConfig())

    # 2. YAML 文件
    yaml_dict = _load_yaml(config_path)

    # 3. 环境变量
    env_dict = _load_env_overrides()

    # 4. 命令行参数
    args = _parse_args()
    args_dict = _args_to_overrides(args)

    # 如果命令行指定了不同的 config 路径，重新加载
    if args.config and args.config != config_path:
        yaml_dict = _load_yaml(args.config)

    # 按优先级合并
    merged = _deep_merge(defaults_dict, yaml_dict)
    merged = _deep_merge(merged, env_dict)
    merged = _deep_merge(merged, args_dict)

    # 处理 models 列表
    models_data = merged.pop("models", [])
    model_list = [ModelConfig(**m) for m in models_data] if models_data else []

    config = AppConfig(
        llama_server=LlamaServerConfig(**merged.get("llama_server", {})),
        models=model_list,
        vram=VramConfig(**merged.get("vram", {})),
        cleanup=CleanupConfig(**merged.get("cleanup", {})),
        concurrency=ConcurrencyConfig(**merged.get("concurrency", {})),
        server=ServerConfig(**merged.get("server", {})),
        logging=LoggingConfig(**merged.get("logging", {})),
    )

    return config
