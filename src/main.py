"""
Llama Guardian 主入口
FastAPI 应用，负责请求代理、显存预判、流式响应等
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .cleanup import CleanupWorker
from .config import AppConfig, load_config
from .gpu_monitor import get_gpu_info, get_total_free_vram
from .server_manager import ServerManager
from .vram_estimator import VramEstimator

logger = logging.getLogger("llama-guardian")

# 全局状态
config: AppConfig = None  # type: ignore
server_manager: ServerManager = None  # type: ignore
vram_estimator: VramEstimator = None  # type: ignore
cleanup_worker: CleanupWorker = None  # type: ignore

# 请求统计
request_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "rejected_503": 0,
    "total_inference_time_ms": 0.0,
}

# 全局启动锁（防止并发启动 llama-server）
_start_lock = asyncio.Lock()


def setup_logging(config: AppConfig):
    """配置日志"""
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" if config.logging.format == "text" else None,
    )

    if config.logging.format == "json":
        try:
            from pythonjsonlogger import json as json_logger

            handler = logging.StreamHandler()
            formatter = json_logger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")
            handler.setFormatter(formatter)
            logging.root.handlers = [handler]
        except ImportError:
            logger.warning("python-json-logger 未安装，使用纯文本日志格式")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global config, server_manager, vram_estimator, cleanup_worker

    # 加载配置
    config = load_config()
    setup_logging(config)

    logger.info("Llama Guardian 启动中...")
    logger.info(f"配置: server={config.server.host}:{config.server.port}, llama-server={config.llama_server.host}:{config.llama_server.port}")

    # 初始化组件
    server_manager = ServerManager(config)
    vram_estimator = VramEstimator(config)

    logger.info("并发限制: 透传模式（由 llama-server 自行管理并发）")

    # 初始化清理守护线程
    cleanup_worker = CleanupWorker(
        config=config.cleanup,
        stop_callback=server_manager.stop,
        is_running_callback=lambda: server_manager.is_running,
    )
    cleanup_worker.start()

    # 列出已配置的模型
    models = vram_estimator.list_models()
    for m in models:
        logger.info(f"已配置模型: {m['name']} (路径: {m['path']}, 估算显存: {m['estimated_vram_mb']}MB)")

    logger.info("Llama Guardian 启动完成")

    yield

    # 关闭清理
    logger.info("Llama Guardian 正在关闭...")
    if cleanup_worker:
        cleanup_worker.stop()
    if server_manager and server_manager.is_running:
        await server_manager.stop()
    logger.info("Llama Guardian 已关闭")


app = FastAPI(
    title="Llama Guardian",
    description="GPU 资源调度中间层",
    version="0.1.0",
    lifespan=lifespan,
)


# ============ 请求代理 ============


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
)
async def proxy_v1_request(request: Request, path: str):
    """
    代理 /v1/ 路径下的所有请求到 llama-server
    支持 OpenAI 兼容的 API（/v1/chat/completions, /v1/completions 等）
    """
    return await _handle_proxy(request, path)


async def _handle_proxy(request: Request, path: str):
    """核心代理逻辑"""
    global request_stats

    request_start_time = time.time()
    request_stats["total_requests"] += 1

    # --- 1. 解析请求中的 model 字段 ---
    body = await request.body()
    body_json = {}
    if body:
        try:
            body_json = json.loads(body)
        except Exception:
            pass

    model_name = body_json.get("model", "")

    # 如果请求中没有 model 字段，尝试使用默认模型
    if not model_name and config.llama_server.default_model_path:
        # 使用默认模型（从配置中查找名称）
        for m in config.models:
            if m.path == config.llama_server.default_model_path:
                model_name = m.name
                break

    if not model_name:
        request_stats["failed_requests"] += 1
        raise HTTPException(status_code=400, detail="请求中缺少 'model' 字段，且未配置默认模型")

    # --- 2. 检查模型是否存在 ---
    model_path = vram_estimator.get_model_path(model_name)
    if not model_path:
        request_stats["failed_requests"] += 1
        raise HTTPException(status_code=404, detail=f"未找到模型: {model_name}。可用模型: {[m.name for m in config.models]}")

    # --- 3. 显存预判 ---
    required_vram = vram_estimator.estimate_required_vram(model_name)
    if required_vram < 0:
        request_stats["failed_requests"] += 1
        raise HTTPException(status_code=500, detail=f"无法估算模型 {model_name} 的显存需求")

    available_vram = get_total_free_vram()

    # 如果 llama-server 已在运行且加载了同一模型，不重复检查显存
    if not (server_manager.is_running and server_manager.current_model == model_name):
        if available_vram < required_vram:
            request_stats["rejected_503"] += 1
            logger.warning(f"显存不足: 需要 ~{required_vram}MB, 可用 {available_vram}MB")
            raise HTTPException(status_code=503, detail=f"Insufficient VRAM. Required: ~{required_vram} MB, Available: {available_vram} MB.")

    # --- 4. 确保 llama-server 运行 ---
    async with _start_lock:
        if not server_manager.is_running or server_manager.current_model != model_name:
            success = await server_manager.start(model_name, model_path)
            if not success:
                request_stats["failed_requests"] += 1
                raise HTTPException(status_code=503, detail="Failed to start llama-server. VRAM fragmentation may have caused OOM.")

    # --- 5. 更新活跃时间 ---
    cleanup_worker.touch()

    # --- 6. 转发请求到 llama-server ---
    is_stream = body_json.get("stream", False)
    target_url = f"{server_manager.base_url}/v1/{path}"

    try:
        if is_stream:
            return await _stream_proxy(request, target_url, body)
        else:
            return await _normal_proxy(request, target_url, body)
    except httpx.ConnectError:
        request_stats["failed_requests"] += 1
        raise HTTPException(status_code=503, detail="Backend llama-server is unreachable. It may have crashed during inference.")
    except httpx.TimeoutException:
        request_stats["failed_requests"] += 1
        raise HTTPException(status_code=504, detail="Backend llama-server request timed out.")
    finally:
        elapsed_ms = (time.time() - request_start_time) * 1000
        request_stats["total_inference_time_ms"] += elapsed_ms
        request_stats["successful_requests"] += 1


async def _normal_proxy(request: Request, target_url: str, body: bytes) -> JSONResponse:
    """非流式请求代理"""
    async with httpx.AsyncClient(timeout=300.0) as client:
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        resp = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body if isinstance(body, bytes) else body.encode(),
        )
        return JSONResponse(
            content=resp.json(),
            status_code=resp.status_code,
        )


async def _stream_proxy(request: Request, target_url: str, body: bytes) -> StreamingResponse:
    """流式请求代理（SSE）"""

    async def stream_generator():
        async with httpx.AsyncClient(timeout=300.0) as client:
            headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
            async with client.stream(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body if isinstance(body, bytes) else body.encode(),
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============ Guardian 管理端点 ============


@app.get("/health")
async def health():
    """健康检查 + GPU 显存状态"""
    gpus = get_gpu_info()
    gpu_dicts = [g.to_dict() for g in gpus]
    total_free = sum(g.memory_free_mb for g in gpus)

    return {
        "status": "healthy",
        "llama_server_running": server_manager.is_running if server_manager else False,
        "current_model": server_manager.current_model if server_manager else None,
        "gpus": gpu_dicts,
        "total_free_vram_mb": total_free,
        "uptime_seconds": server_manager.uptime_seconds if server_manager else None,
    }


@app.get("/metrics")
async def metrics():
    """请求统计"""
    total = request_stats["total_requests"]
    avg_latency = request_stats["total_inference_time_ms"] / total if total > 0 else 0

    return {
        "total_requests": total,
        "successful_requests": request_stats["successful_requests"],
        "failed_requests": request_stats["failed_requests"],
        "rejected_503": request_stats["rejected_503"],
        "avg_latency_ms": round(avg_latency, 2),
        "total_inference_time_ms": round(request_stats["total_inference_time_ms"], 2),
        "llama_server_running": server_manager.is_running if server_manager else False,
        "current_model": server_manager.current_model if server_manager else None,
        "last_active_time": cleanup_worker.last_active_time if cleanup_worker else None,
        "models": vram_estimator.list_models() if vram_estimator else [],
    }


@app.get("/status")
async def status():
    """详细的运行状态"""
    result = {
        "guardian": {
            "version": "0.1.0",
            "server": f"{config.server.host}:{config.server.port}" if config else None,
        },
        "llama_server": server_manager.get_status() if server_manager else None,
        "gpu": {
            "gpus": [g.to_dict() for g in get_gpu_info()],
            "total_free_vram_mb": get_total_free_vram(),
        },
        "config": {
            "idle_timeout_seconds": config.cleanup.idle_timeout_seconds if config else None,
            "max_concurrent_requests": config.concurrency.max_concurrent_requests if config else None,
            "vram_size_multiplier": config.vram.size_multiplier if config else None,
            "safety_margin_mb": config.vram.safety_margin_mb if config else None,
        },
    }

    if server_manager and server_manager.is_running:
        llama_health = await server_manager.health_check()
        result["llama_server"]["health"] = llama_health

    return result


def main():
    """主入口"""
    global config
    config = load_config()
    setup_logging(config)

    uvicorn.run(
        "src.main:app",
        host=config.server.host,
        port=config.server.port,
        log_level=config.logging.level.lower(),
    )


if __name__ == "__main__":
    main()
