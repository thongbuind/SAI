"""Ứng dụng FastAPI của web SAI bản server.

- ``/api/*``: chat, mô hình, lịch sử, admin (analytics).
- ``/api/lab/*``: phòng làm việc, chỉ khi truyền ``lab`` (``python -m web --lab``); không bật thì trả 503.
- ``/admin/``: trang admin đọc ``/api/admin/*``.
- ``/``: site Astro đã build (``web/site/dist/server``). ``StaticFiles(html=True)`` trả ``index.html`` của
  thư mục, ``404.html`` khi không có trang và chặn đường dẫn thoát ra ngoài thư mục site.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from web.backend.analytics_middleware import AnalyticsMiddleware, init_db
from web.backend.analytics_routes import router as analytics_router
from web.backend.chat_routes import RATE_LIMIT_SEC, RateLimiter, router as chat_router
from web.backend.history_routes import router as history_router
from web.backend.lab_routes import Lab, disabled_router as lab_disabled_router, router as lab_router
from web.backend.model_registry import ModelRegistry
from web.backend.paths import ADMIN_PAGE

API_PREFIX = "/api"
LAB_PREFIX = "/api/lab"

def create_app(site_dir: Path, registry: ModelRegistry, lab: Lab | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db()
        registry.start()
        if lab is not None:
            lab.runner.start()
        yield
        if lab is not None:
            await lab.runner.stop()
        await registry.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.registry = registry
    app.state.lab = lab
    app.state.rate_limiter = RateLimiter(RATE_LIMIT_SEC)
    app.add_middleware(AnalyticsMiddleware)
    for router in (chat_router, history_router, analytics_router):
        app.include_router(router, prefix=API_PREFIX)
    app.include_router(lab_router if lab is not None else lab_disabled_router, prefix=LAB_PREFIX)

    @app.get("/admin/", include_in_schema=False)
    def admin_page():
        return FileResponse(ADMIN_PAGE)

    # Mount sau cùng để mọi route ở trên được khớp trước.
    app.mount("/", StaticFiles(directory=site_dir, html=True), name="site")
    return app
