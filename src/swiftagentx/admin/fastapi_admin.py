"""
FastAPI Router for the admin API.

Usage::

    from swiftagentx.admin import AdminService, create_fastapi_admin_router

    service = AdminService(agent)
    router = create_fastapi_admin_router(service)
    app.include_router(router, prefix="/admin")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .service import AdminService


def create_fastapi_admin_router(
    service: "AdminService",
    prefix: str = "/admin",
    tags: Optional[List[str]] = None,
):
    """
    Create a FastAPI APIRouter wired to *service*.

    Returns a ``fastapi.APIRouter`` — include it in your FastAPI app.
    """
    try:
        from fastapi import APIRouter
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("FastAPI is required for the admin router: pip install fastapi")

    router = APIRouter(prefix=prefix, tags=tags or ["admin"])

    # -- Request models --

    class CacheClearRequest(BaseModel):
        level: Optional[str] = None

    class KBSearchRequest(BaseModel):
        query: str
        top_k: int = 5

    class KBAddDocumentsRequest(BaseModel):
        documents: List[Dict[str, Any]]

    class ConfigUpdateRequest(BaseModel):
        updates: Dict[str, Any] = {}

    # -- Endpoints --

    @router.get("/status")
    async def status():
        return service.get_status()

    @router.get("/tools")
    async def tools():
        return service.get_tools()

    @router.get("/cache/stats")
    async def cache_stats():
        return service.get_cache_stats()

    @router.post("/cache/clear")
    async def cache_clear(body: CacheClearRequest):
        return service.clear_cache(body.level)

    @router.get("/config")
    async def get_config():
        return service.get_config()

    @router.put("/config")
    async def update_config(body: ConfigUpdateRequest):
        return service.update_config(body.updates)

    @router.post("/kb/search")
    async def kb_search(body: KBSearchRequest):
        return await service.kb_search_async(body.query, top_k=body.top_k)

    @router.post("/kb/documents")
    async def kb_add_documents(body: KBAddDocumentsRequest):
        return await service.kb_add_documents_async(body.documents)

    @router.delete("/kb/documents/{doc_id}")
    async def kb_delete_document(doc_id: str):
        return await service.kb_delete_document_async(doc_id)

    @router.get("/kb/stats")
    async def kb_stats():
        return await service.kb_stats_async()

    return router
