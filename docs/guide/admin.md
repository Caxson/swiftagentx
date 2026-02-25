# 管理后台指南

## 概述

SwiftAgent 提供框架无关的管理服务层 `AdminService`，以及 Flask Blueprint 和 FastAPI Router 两套薄路由层。

## 快速开始

### Flask

```python
from flask import Flask
from swiftagentx import Agent
from swiftagentx.admin import AdminService, create_flask_admin_blueprint

app = Flask(__name__)
agent = Agent(model=my_model)
service = AdminService(agent)

bp = create_flask_admin_blueprint(service)
app.register_blueprint(bp, url_prefix="/admin")
```

### FastAPI

```python
from fastapi import FastAPI
from swiftagentx import Agent
from swiftagentx.admin import AdminService, create_fastapi_admin_router

app = FastAPI()
agent = Agent(model=my_model)
service = AdminService(agent)

router = create_fastapi_admin_router(service)
app.include_router(router, prefix="/admin")
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/admin/status` | Agent 状态（名称、工具数、缓存统计、运行时间） |
| GET | `/admin/tools` | 已注册工具列表及 JSON Schema |
| GET | `/admin/cache/stats` | 缓存统计 |
| POST | `/admin/cache/clear` | 清除缓存 (body: `{"level": "level_1"}` 或空) |
| GET | `/admin/config` | 获取配置（敏感值脱敏） |
| PUT | `/admin/config` | 更新配置 (body: `{"max_iterations": 20}`) |
| POST | `/admin/kb/search` | 知识库搜索 (body: `{"query": "...", "top_k": 5}`) |
| POST | `/admin/kb/documents` | 添加文档 (body: `{"documents": [...]}`) |
| DELETE | `/admin/kb/documents/:id` | 删除文档 |
| GET | `/admin/kb/stats` | 知识库统计 |

## AdminService

核心逻辑层，不依赖任何 Web 框架：

```python
class AdminService:
    def __init__(self, agent: Agent): ...

    # 状态查询
    def get_status(self) -> Dict: ...
    def get_tools(self) -> List[Dict]: ...
    def get_cache_stats(self) -> Dict: ...
    def get_config(self) -> Dict: ...

    # 配置管理
    def update_config(self, updates: Dict) -> Dict: ...

    # 缓存管理
    def clear_cache(self, level: Optional[str] = None) -> Dict: ...

    # 知识库管理 (async 版本)
    async def kb_search_async(self, query: str, top_k: int = 5) -> List[Dict]: ...
    async def kb_add_documents_async(self, documents: List[Dict]) -> Dict: ...
    async def kb_delete_document_async(self, doc_id: str) -> Dict: ...
    async def kb_stats_async(self) -> Dict: ...
```

## 配置脱敏

`get_config()` 自动对包含 `key`、`secret`、`token`、`password`、`credential` 的字段进行脱敏：

```json
{
    "name": "MyAgent",
    "api_key": "sk-1***"
}
```

## 安全提示

Admin API **不内置认证**。生产环境中请添加：

- Flask: 使用 `@login_required` 装饰器或自定义中间件
- FastAPI: 使用 `Depends()` 注入认证依赖

```python
# Flask 示例
from functools import wraps
from flask import request, abort

def require_admin_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("X-Admin-Token")
        if token != "your-secret-token":
            abort(403)
        return f(*args, **kwargs)
    return decorated

# 注册 Blueprint 后添加认证
for rule in bp.deferred_functions:
    # 或者在 Blueprint 的 before_request 中统一检查
    pass

@bp.before_request
def check_admin_auth():
    token = request.headers.get("X-Admin-Token")
    if token != os.environ.get("ADMIN_TOKEN"):
        abort(403)
```

```python
# FastAPI 示例
from fastapi import Depends, HTTPException, Header

async def verify_admin(x_admin_token: str = Header(...)):
    if x_admin_token != os.environ.get("ADMIN_TOKEN"):
        raise HTTPException(status_code=403, detail="Forbidden")

router = create_fastapi_admin_router(service)
# 全局依赖
app.include_router(router, prefix="/admin", dependencies=[Depends(verify_admin)])
```
