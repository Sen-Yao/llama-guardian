"""
llama-server 进程管理模块
负责启动、停止、健康检查 llama-server 进程
"""

import asyncio
import os
import time
import logging
import subprocess
from typing import Optional

import httpx

from .config import AppConfig
from .gpu_monitor import get_gpu_count

logger = logging.getLogger("llama-guardian")


class ServerManager:
    """管理 llama-server 进程的生命周期"""

    def __init__(self, config: AppConfig):
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._current_model: Optional[str] = None
        self._start_time: Optional[float] = None
        self._base_url = f"http://{config.llama_server.host}:{config.llama_server.port}"

    @property
    def is_running(self) -> bool:
        """llama-server 进程是否在运行"""
        return self._process is not None and self._process.poll() is None

    @property
    def current_model(self) -> Optional[str]:
        """当前加载的模型名称"""
        return self._current_model

    @property
    def uptime_seconds(self) -> Optional[float]:
        """进程运行时长（秒）"""
        if self._start_time is None:
            return None
        return time.time() - self._start_time

    @property
    def base_url(self) -> str:
        """llama-server 的基础 URL"""
        return self._base_url

    def _build_command(self, model_path: str) -> list[str]:
        """
        构建 llama-server 启动命令

        使用 --split-mode row 进行张量并行切分
        --n-gpu-layers 999 强制所有层加载到 GPU
        """
        cfg = self.config.llama_server
        gpu_count = get_gpu_count()

        cmd = [
            cfg.binary_path,
            "-m", model_path,
            "--split-mode", "row",
            "--n-gpu-layers", "999",
            "-c", str(cfg.context_size),
            "--host", cfg.host,
            "--port", str(cfg.port),
        ]

        # 如果有多个 GPU，使用 --tensor-split 按剩余显存比例分配
        # （llama.cpp 的 --split-mode row 已能自动处理，此处留作扩展）
        if gpu_count > 0:
            logger.info(f"检测到 {gpu_count} 张 GPU，将使用 --split-mode row 进行张量并行")

        # 添加额外参数
        if cfg.extra_args:
            cmd.extend(cfg.extra_args)

        return cmd

    async def start(self, model_name: str, model_path: str) -> bool:
        """
        启动 llama-server 进程

        Args:
            model_name: 模型名称
            model_path: 模型文件路径

        Returns:
            是否启动成功
        """
        # 如果已经在运行，先停止
        if self.is_running:
            if self._current_model == model_name:
                logger.info(f"llama-server 已在运行模型 {model_name}，无需重启")
                return True
            else:
                logger.info(f"切换模型: {self._current_model} -> {model_name}，先停止当前服务")
                await self.stop()

        cmd = self._build_command(model_path)
        logger.info(f"启动 llama-server: {' '.join(cmd)}")

        try:
            # 构建 env，确保 LD_LIBRARY_PATH 包含 llama-server 的库目录
            env = os.environ.copy()
            bin_dir = os.path.dirname(self.config.llama_server.binary_path)
            ld_path = env.get("LD_LIBRARY_PATH", "")
            if bin_dir not in ld_path:
                env["LD_LIBRARY_PATH"] = f"{bin_dir}:{ld_path}".rstrip(":")

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError:
            logger.error(f"llama-server 可执行文件未找到: {self.config.llama_server.binary_path}")
            return False
        except Exception as e:
            logger.error(f"启动 llama-server 失败: {e}")
            self._process = None
            return False

        # 等待进程启动，检查是否立即退出（OOM 等）
        await asyncio.sleep(3)

        if self._process.poll() is not None:
            # 进程已退出，读取 stderr 获取错误信息
            stderr_output = ""
            try:
                stderr_output = self._process.stderr.read().decode("utf-8", errors="replace")[-500:]
            except Exception:
                pass
            logger.error(f"llama-server 启动后立即退出 (code={self._process.returncode}): {stderr_output}")
            self._process = None
            return False

        # 等待 HTTP 服务就绪（健康检查）
        ready = await self._wait_for_ready(max_retries=30, interval=1.0)
        if not ready:
            logger.error("llama-server 启动超时，HTTP 服务未就绪")
            await self.stop()
            return False

        self._current_model = model_name
        self._start_time = time.time()
        logger.info(f"llama-server 启动成功，模型: {model_name}")
        return True

    async def stop(self):
        """停止 llama-server 进程"""
        if self._process is None:
            return

        logger.info("正在停止 llama-server...")

        # 先尝试优雅终止
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server 未能在 10 秒内优雅退出，强制终止")
            self._process.kill()
            self._process.wait(timeout=5)

        self._process = None
        self._current_model = None
        self._start_time = None
        logger.info("llama-server 已停止")

    async def _wait_for_ready(self, max_retries: int = 30, interval: float = 1.0) -> bool:
        """
        等待 llama-server HTTP 服务就绪

        通过请求 /health 端点检测
        """
        for i in range(max_retries):
            if self._process.poll() is not None:
                return False
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{self._base_url}/health", timeout=2.0)
                    if resp.status_code == 200:
                        return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            await asyncio.sleep(interval)

        return False

    async def health_check(self) -> dict:
        """
        检查 llama-server 健康状态

        Returns:
            包含健康状态信息的字典
        """
        result = {
            "process_running": self.is_running,
            "current_model": self._current_model,
            "uptime_seconds": self.uptime_seconds,
            "http_ready": False,
        }

        if self.is_running:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{self._base_url}/health", timeout=5.0)
                    result["http_ready"] = resp.status_code == 200
                    result["llama_server_response"] = resp.json()
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                result["http_error"] = str(e)

        return result

    def get_status(self) -> dict:
        """获取当前状态摘要"""
        return {
            "is_running": self.is_running,
            "current_model": self._current_model,
            "uptime_seconds": self.uptime_seconds,
            "pid": self._process.pid if self._process else None,
            "base_url": self._base_url,
        }