"""
FastAPI Router for the admin API.

Usage::

    from swiftagentx.admin import AdminService, create_fastapi_admin_router

    service = AdminService(agent)
    app.include_router(create_fastapi_admin_router(service), prefix="/admin")

The router itself no longer adds a prefix — let ``include_router`` decide
where to mount it. This avoids the double-prefix bug ("/admin/admin/status")
that bit v0.3.0 users.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .service import AdminService


def create_fastapi_admin_router(
    service: AdminService,
    prefix: str = "",
    tags: list[str] | None = None,
):
    """
    Create a FastAPI APIRouter wired to *service*.

    Args:
        service: The :class:`AdminService` to expose.
        prefix: Prefix to bake into the router. Defaults to ``""`` so the
            user can supply the mount path via
            ``app.include_router(router, prefix="/admin")`` — that is the
            FastAPI idiom and matches the README pattern. Passing a
            non-empty prefix here is supported for symmetry with
            ``create_flask_admin_blueprint`` but no longer the
            recommended path.
        tags: OpenAPI tags applied to every route.

    Returns a ``fastapi.APIRouter`` — include it in your FastAPI app.
    """
    try:
        from fastapi import APIRouter
        from pydantic import BaseModel
    except ImportError as exc:
        raise ImportError("FastAPI is required for the admin router: pip install fastapi") from exc

    router = APIRouter(prefix=prefix, tags=tags or ["admin"])

    # -- Request models --

    class CacheClearRequest(BaseModel):
        level: str | None = None

    class KBSearchRequest(BaseModel):
        query: str
        top_k: int = 5

    class KBAddDocumentsRequest(BaseModel):
        documents: list[dict[str, Any]]

    class ConfigUpdateRequest(BaseModel):
        updates: dict[str, Any] = {}

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
