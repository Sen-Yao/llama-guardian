"""
GPU 显存监控模块
通过 nvidia-smi 实时获取各显卡的显存信息
"""

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger("llama-guardian")


@dataclass
class GPUInfo:
    """单张 GPU 的信息"""
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "memory_total_mb": self.memory_total_mb,
            "memory_used_mb": self.memory_used_mb,
            "memory_free_mb": self.memory_free_mb,
        }


def get_gpu_info() -> list[GPUInfo]:
    """
    获取所有 GPU 的显存信息

    Returns:
        GPUInfo 列表，每个元素代表一张显卡
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        result.check_returncode()
    except FileNotFoundError:
        logger.error("nvidia-smi 未找到，请确保 NVIDIA 驱动已安装")
        return []
    except subprocess.TimeoutExpired:
        logger.error("nvidia-smi 执行超时")
        return []
    except subprocess.CalledProcessError as e:
        logger.error(f"nvidia-smi 执行失败: {e.stderr}")
        return []

    gpus = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            logger.warning(f"无法解析 nvidia-smi 输出行: {line}")
            continue
        try:
            gpu = GPUInfo(
                index=int(parts[0]),
                name=parts[1],
                memory_total_mb=int(float(parts[2])),
                memory_used_mb=int(float(parts[3])),
                memory_free_mb=int(float(parts[4])),
            )
            gpus.append(gpu)
        except (ValueError, IndexError) as e:
            logger.warning(f"解析 GPU 信息失败: {line}, 错误: {e}")

    return gpus


def get_total_free_vram() -> int:
    """
    获取所有 GPU 的总剩余显存（MB）

    Returns:
        总剩余显存 MB 数
    """
    gpus = get_gpu_info()
    return sum(gpu.memory_free_mb for gpu in gpus)


def get_gpu_count() -> int:
    """获取 GPU 数量"""
    return len(get_gpu_info())