"""
Admin module — management API for SwiftAgent.
"""

from .fastapi_admin import create_fastapi_admin_router
from .flask_admin import create_flask_admin_blueprint
from .service import AdminService

__all__ = [
    "AdminService",
    "create_flask_admin_blueprint",
    "create_fastapi_admin_router",
]
