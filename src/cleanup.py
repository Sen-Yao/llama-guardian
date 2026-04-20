"""
闲置清理模块
后台定期检查，超时未使用则自动关闭 llama-server 释放显存
"""

import asyncio
import logging
import threading
import time
from typing import Awaitable, Callable, Optional

from .config import CleanupConfig

logger = logging.getLogger("llama-guardian")


class CleanupWorker:
    """闲置清理守护线程"""

    def __init__(
        self,
        config: CleanupConfig,
        stop_callback: Callable[[], Awaitable[None]],
        is_running_callback: Callable[[], bool],
    ):
        """
        Args:
            config: 清理配置
            stop_callback: 异步停止函数（用于停止 llama-server）
            is_running_callback: 检查 llama-server 是否在运行的函数
        """
        self.config = config
        self._stop_callback = stop_callback
        self._is_running_callback = is_running_callback
        self._last_active_time: float = time.time()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def last_active_time(self) -> float:
        """最后活跃时间戳"""
        return self._last_active_time

    def touch(self):
        """更新最后活跃时间（收到新请求时调用）"""
        self._last_active_time = time.time()

    def start(self):
        """启动清理守护线程"""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="cleanup-worker")
        self._thread.start()
        logger.info(f"清理守护线程已启动，闲置超时={self.config.idle_timeout_seconds}秒，检查间隔={self.config.check_interval_seconds}秒")

    def stop(self):
        """停止清理守护线程"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self):
        """守护线程主循环"""
        # 为这个线程创建一个事件循环，用于调用异步 stop_callback
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        while not self._stop_event.is_set():
            self._stop_event.wait(self.config.check_interval_seconds)
            if self._stop_event.is_set():
                break

            if not self._is_running_callback():
                continue

            idle_seconds = time.time() - self._last_active_time
            if idle_seconds > self.config.idle_timeout_seconds:
                logger.info(f"llama-server 已闲置 {idle_seconds:.0f} 秒 (阈值: {self.config.idle_timeout_seconds}秒)，正在关闭...")
                try:
                    self._loop.run_until_complete(self._stop_callback())
                    logger.info("闲置清理完成，llama-server 已关闭，显存已释放")
                except Exception as e:
                    logger.error(f"闲置清理失败: {e}")

        self._loop.close()
