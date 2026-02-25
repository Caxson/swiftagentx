# 生产部署指南

## 概述

SwiftAgent 支持 Flask 和 FastAPI 两种 Web 框架部署。本文档介绍生产环境的推荐配置。

## Gunicorn (Flask)

```bash
gunicorn -w 4 --threads 4 -b 0.0.0.0:5000 \
    --timeout 1800 \
    --log-level info \
    --access-logfile gunicorn_access.log \
    --error-logfile gunicorn_error.log \
    app:app
```

参数说明：
- `-w 4`: Worker 数量（建议 CPU 核数 × 2 + 1）
- `--threads 4`: 每个 Worker 的线程数
- `--timeout 1800`: Agent 处理可能耗时，设置较长超时

## Uvicorn (FastAPI)

```bash
uvicorn app:app --host 0.0.0.0 --port 5000 \
    --workers 4 \
    --timeout-keep-alive 120 \
    --log-level info
```

## Docker 部署

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Flask
CMD ["gunicorn", "-w", "4", "--threads", "4", "-b", "0.0.0.0:5000", "--timeout", "1800", "app:app"]

# FastAPI
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "4"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  agent:
    build: .
    ports:
      - "5000:5000"
    environment:
      - MODEL_API_KEY=${MODEL_API_KEY}
      - MODEL_NAME=qwen-turbo
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Nginx 反向代理

```nginx
upstream agent_backend {
    server 127.0.0.1:5000;
    keepalive 32;
}

server {
    listen 80;
    server_name agent.example.com;

    # SSE 需要关闭缓冲
    location /api/v1/agent/sse {
        proxy_pass http://agent_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 1800s;
        chunked_transfer_encoding on;
    }

    location / {
        proxy_pass http://agent_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

关键配置：
- `proxy_buffering off` — SSE 必须关闭缓冲
- `proxy_read_timeout 1800s` — Agent 处理超时
- `keepalive` — 减少连接开销

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MODEL_API_KEY` | LLM API Key | (必填) |
| `MODEL_NAME` | 模型名称 | `qwen-turbo` |
| `MODEL_BASE_URL` | API 地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `ENABLE_CACHE` | 启用缓存 | `true` |
| `MAX_ITERATIONS` | 最大迭代次数 | `10` |

## 健康检查

建议在应用中添加健康检查端点：

```python
@app.route("/health")
def health():
    return {"status": "ok", "agent": agent.name}
```

## 监控

通过 Admin API 监控运行状态：

```bash
# 查看状态
curl http://localhost:5000/admin/status

# 查看缓存
curl http://localhost:5000/admin/cache/stats

# 清空缓存
curl -X POST http://localhost:5000/admin/cache/clear
```

## 日志配置

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# 降低第三方库日志级别
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
```
