"""
GPU 监控模块测试
注意：在无 GPU 环境下，部分测试需要 mock nvidia-smi
"""

from unittest.mock import MagicMock, patch

from src.gpu_monitor import GPUInfo, get_gpu_count, get_gpu_info, get_total_free_vram


def test_gpu_info_to_dict():
    """测试 GPUInfo 数据类"""
    gpu = GPUInfo(
        index=0,
        name="NVIDIA L40",
        memory_total_mb=46068,
        memory_used_mb=8000,
        memory_free_mb=38068,
    )
    d = gpu.to_dict()
    assert d["index"] == 0
    assert d["name"] == "NVIDIA L40"
    assert d["memory_total_mb"] == 46068
    assert d["memory_free_mb"] == 38068


@patch("src.gpu_monitor.subprocess.run")
def test_get_gpu_info(mock_run):
    """测试获取 GPU 信息（mock nvidia-smi）"""
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="0, NVIDIA L40, 46068, 8000, 38068\n1, NVIDIA L40, 46068, 5000, 41068\n",
    )

    gpus = get_gpu_info()
    assert len(gpus) == 2
    assert gpus[0].index == 0
    assert gpus[0].name == "NVIDIA L40"
    assert gpus[0].memory_free_mb == 38068
    assert gpus[1].index == 1
    assert gpus[1].memory_free_mb == 41068


@patch("src.gpu_monitor.subprocess.run")
def test_get_total_free_vram(mock_run):
    """测试获取总剩余显存"""
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="0, NVIDIA L40, 46068, 8000, 38068\n1, NVIDIA L40, 46068, 5000, 41068\n",
    )

    total = get_total_free_vram()
    assert total == 38068 + 41068


@patch("src.gpu_monitor.subprocess.run")
def test_get_gpu_count(mock_run):
    """测试获取 GPU 数量"""
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="0, NVIDIA L40, 46068, 8000, 38068\n1, NVIDIA L40, 46068, 5000, 41068\n",
    )

    count = get_gpu_count()
    assert count == 2


@patch("src.gpu_monitor.subprocess.run")
def test_get_gpu_info_no_nvidia_smi(mock_run):
    """测试 nvidia-smi 不可用的情况"""
    mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

    gpus = get_gpu_info()
    assert gpus == []
