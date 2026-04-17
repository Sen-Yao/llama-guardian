# 🦙 Llama Guardian

**GPU 资源调度中间层** — 在共享 GPU 服务器上智能调度 llama-server，实现按需启停、显存预判、张量并行切分。

## 概述

Llama Guardian 是一个轻量级的 GPU 显卡资源调度器，部署在上游 LLM 网关和 llama.cpp 之间，解决公共服务器上 GPU 显存争用的问题。

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  上游 LLM     │────▶│  Llama Guardian  │────▶│  llama-server │
│  网关 / 客户端 │◀────│  (FastAPI)       │◀────│  (llama.cpp)  │
└──────────────┘     └────────┬─────────┘     └──────────────┘
                              │
                     ┌────────▼─────────┐
                     │   nvidia-smi      │
                     │   8 × L40 GPU    │
                     └──────────────────┘
```

### 核心特性

- **显存预判**：请求到达时自动估算模型所需显存，不足则直接返回 503
- **张量并行**：根据各显卡剩余显存，自动配置 `--split-mode row` 实现跨卡切分
- **按需启停**：首次请求自动拉起 llama-server，闲置超时后自动关闭释放显存
- **OpenAI 兼容**：完整支持 `/v1/chat/completions` 等 OpenAI API 格式，含 SSE 流式响应
- **动态模型切换**：通过请求中的 `model` 字段指定模型，自动切换（需重启 llama-server）
- **零侵入**：纯 Python 实现，无需 Docker，无需 root 权限

## 适用场景

- 公共 GPU 服务器（多用户共享显卡资源）
- 显存碎片化严重，无法保证单卡完整性
- 需要为其他程序（训练、数据处理等）预留 GPU 资源
- 间歇性 LLM 推理需求，不想长期占用显存

## 快速开始

### 前置条件

- Python 3.10+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) 已编译（需要 `llama-server` 可执行文件）
- NVIDIA 驱动 + nvidia-smi 可用
- 至少 1 张 NVIDIA GPU

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourname/llama-guardian.git
cd llama-guardian

# 安装依赖
pip install -r requirements.txt
```

### 配置

```bash
# 复制配置模板
cp config.example.yaml config.yaml

# 编辑配置（至少修改 llama_server.binary_path 和模型路径）
vim config.yaml
```

### 启动

```bash
# 前台运行（开发/调试）
python -m src.main

# 或指定配置文件
python -m src.main --config /path/to/config.yaml

# 或通过环境变量覆盖
export LLAMA_SERVER__BINARY_PATH=/usr/local/bin/llama-server
export SERVER__PORT=8080
python -m src.main
```

## 配置说明

配置优先级：**命令行参数 > 环境变量 > config.yaml > 默认值**

### config.yaml 完整示例

```yaml
# llama-server 配置
llama_server:
  binary_path: "./llama-server"         # llama-server 可执行文件路径
  host: "127.0.0.1"                     # llama-server 监听地址
  port: 11434                           # llama-server 监听端口
  default_model_path: null              # 默认模型路径（可选）
  context_size: 4096                    # 上下文窗口大小
  extra_args: []                        # 额外的 llama-server 参数

# 显存估算
vram:
  # 模型文件大小 × 此倍率 = 预估所需显存（含 KV cache、上下文等开销）
  size_multiplier: 1.4
  # 安全边际：额外保留的显存量（MB），给其他程序留空间
  safety_margin_mb: 2000

# 闲置清理
cleanup:
  idle_timeout_seconds: 300             # 闲置 N 秒后关闭 llama-server
  check_interval_seconds: 10            # 检查间隔

# 并发控制
concurrency:
  max_concurrent_requests: 0            # 最大并发数，0 = 不限流（纯透传）

# Guardian 服务器配置
server:
  host: "0.0.0.0"                       # Guardian 监听地址
  port: 8000                            # Guardian 监听端口

# 日志配置
logging:
  level: "INFO"                         # DEBUG / INFO / WARNING / ERROR
  format: "json"                        # json 或 text
```

### 环境变量

环境变量使用双下划线 `__` 表示层级，全大写。例如：

```bash
# 对应 config.yaml 中的 llama_server.binary_path
export LLAMA_SERVER__BINARY_PATH=/usr/local/bin/llama-server

# 对应 vram.size_multiplier
export VRAM__SIZE_MULTIPLIER=1.5

# 对应 cleanup.idle_timeout_seconds
export CLEANUP__IDLE_TIMEOUT_SECONDS=600
```

### 命令行参数

```bash
python -m src.main --help

# 常用参数
python -m src.main \
  --config ./config.yaml \
  --server.port 8080 \
  --llama-server.binary-path /usr/local/bin/llama-server \
  --cleanup.idle-timeout-seconds 600
```

## API 端点

### LLM 代理（透传至 llama-server）

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | OpenAI 兼容的聊天补全（支持 SSE 流式） |
| `/v1/completions` | POST | OpenAI 兼容的文本补全 |
| `/v1/models` | GET | 列出可用模型 |

### Guardian 管理

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 + GPU 显存状态 |
| `/metrics` | GET | 请求统计（请求数、成功率、延迟等） |
| `/status` | GET | llama-server 运行状态详情 |

### 响应示例

**GET /health**

```json
{
  "status": "healthy",
  "llama_server_running": true,
  "current_model": "Qwen2.5-72B-Q4_K_M",
  "gpus": [
    { "index": 0, "name": "NVIDIA L40", "memory_total_mb": 46068, "memory_free_mb": 38420 },
    { "index": 1, "name": "NVIDIA L40", "memory_total_mb": 46068, "memory_free_mb": 41200 }
  ],
  "total_free_vram_mb": 79620,
  "uptime_seconds": 1847
}
```

**显存不足时的错误响应**

```json
{
  "detail": "Insufficient VRAM. Required: ~28000 MB, Available: 15200 MB.",
  "status_code": 503
}
```

## 工作原理

### 请求处理流程

```
                    ┌─────────────────────┐
                    │   请求到达          │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  解析 model 字段    │
                    │  确定目标模型       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  查询模型文件大小   │
                    │  × 1.4 = 预估显存   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐     ┌──────────┐
                    │  总剩余显存够不够？ │──否─▶│ 返回 503 │
                    └──────────┬──────────┘     └──────────┘
                               │ 是
                    ┌──────────▼──────────┐
                    │  llama-server 在跑？│
                    │  且模型一致？       │
                    └──────────┬──────────┘
                          │           │
                         否           是
                          │           │
               ┌──────────▼──┐   ┌───▼───────────────┐
               │ 启动/重启    │   │ 透传请求          │
               │ llama-server│   │ StreamingResponse │
               └──────────┬──┘   └───┬───────────────┘
                          │          │
                          └────┬─────┘
                               │
                    ┌──────────▼──────────┐
                    │  更新最后活跃时间   │
                    └─────────────────────┘
```

### 闲置清理流程

后台守护线程每隔 `check_interval_seconds` 检查一次：

```
当前时间 - last_active_time > idle_timeout_seconds ?
    │                │
   是               否
    │                │
终止 llama-server   什么都不做
释放所有显存
```

### 显存估算逻辑

```
所需显存 = 模型文件大小 × size_multiplier + safety_margin_mb
```

- `size_multiplier`（默认 1.4）：覆盖 KV cache、上下文缓冲区、运行时开销
- `safety_margin_mb`（默认 2000 MB）：给其他程序预留的安全边际

## 部署

### 使用 systemd 用户服务（推荐）

无需 root 权限：

```bash
# 1. 复制服务模板
cp llama-guardian.service ~/.config/systemd/user/

# 2. 编辑服务文件中的路径
vim ~/.config/systemd/user/llama-guardian.service

# 3. 加载并启动
systemctl --user daemon-reload
systemctl --user start llama-guardian

# 4. 设置开机自启（需启用 lingering）
loginctl enable-linger $(whoami)
systemctl --user enable llama-guardian

# 5. 查看状态和日志
systemctl --user status llama-guardian
journalctl --user -u llama-guardian -f
```

### 手动后台运行

```bash
nohup python -m src.main --config config.yaml > guardian.log 2>&1 &
```

## 模型配置

模型通过配置文件中的 `models` 部分定义，Guardian 根据请求中的 `model` 字段匹配：

```yaml
models:
  - name: "qwen2.5-72b"
    path: "/data/models/Qwen2.5-72B-Instruct-Q4_K_M.gguf"
  - name: "llama3-70b"
    path: "/data/models/Meta-Llama-3-70B-Instruct-Q4_K_M.gguf"
  - name: "deepseek-v3"
    path: "/data/models/DeepSeek-V3-Q4_K_M.gguf"
```

请求示例：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-72b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

如果请求中的 `model` 未在配置中找到，Guardian 返回 404。

## 技术栈

- **Python 3.10+**
- **FastAPI** — 高性能异步 Web 框架
- **httpx** — 异步 HTTP 客户端（用于代理请求）
- **PyYAML** — YAML 配置解析
- **psutil** — 进程管理

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！