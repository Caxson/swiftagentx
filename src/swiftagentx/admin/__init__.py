"""
Admin module — management API for SwiftAgent.
"""

from .service import AdminService
from .flask_admin import create_flask_admin_blueprint
from .fastapi_admin import create_fastapi_admin_router

__all__ = [
    "AdminService",
    "create_flask_admin_blueprint",
    "create_fastapi_admin_router",
]
