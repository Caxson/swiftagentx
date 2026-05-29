"""
Flask Blueprint for the admin API.

Usage::

    from swiftagentx.admin import AdminService, create_flask_admin_blueprint

    service = AdminService(agent)
    bp = create_flask_admin_blueprint(service)
    app.register_blueprint(bp, url_prefix="/admin")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .service import AdminService


def create_flask_admin_blueprint(
    service: AdminService,
    url_prefix: str = "/admin",
):
    """
    Create a Flask Blueprint wired to *service*.

    Returns a ``flask.Blueprint`` — register it with your Flask app.
    """
    try:
        from flask import Blueprint, jsonify, request
    except ImportError as exc:
        raise ImportError("Flask is required for the admin blueprint: pip install flask") from exc

    bp = Blueprint("swiftagent_admin", __name__, url_prefix=url_prefix)

    @bp.route("/status", methods=["GET"])
    def status():
        return jsonify(service.get_status())

    @bp.route("/tools", methods=["GET"])
    def tools():
        return jsonify(service.get_tools())

    @bp.route("/cache/stats", methods=["GET"])
    def cache_stats():
        return jsonify(service.get_cache_stats())

    @bp.route("/cache/clear", methods=["POST"])
    def cache_clear():
        body = request.get_json(silent=True) or {}
        level = body.get("level")
        return jsonify(service.clear_cache(level))

    @bp.route("/config", methods=["GET"])
    def get_config():
        return jsonify(service.get_config())

    @bp.route("/config", methods=["PUT"])
    def update_config():
        body = request.get_json(silent=True) or {}
        return jsonify(service.update_config(body))

    @bp.route("/kb/search", methods=["POST"])
    def kb_search():
        body = request.get_json(silent=True) or {}
        query = body.get("query", "")
        top_k = body.get("top_k", 5)
        return jsonify(service.kb_search(query, top_k=top_k))

    @bp.route("/kb/documents", methods=["POST"])
    def kb_add_documents():
        body = request.get_json(silent=True) or {}
        documents = body.get("documents", [])
        return jsonify(service.kb_add_documents(documents))

    @bp.route("/kb/documents/<doc_id>", methods=["DELETE"])
    def kb_delete_document(doc_id: str):
        return jsonify(service.kb_delete_document(doc_id))

    @bp.route("/kb/stats", methods=["GET"])
    def kb_stats():
        return jsonify(service.kb_stats())

    return bp
