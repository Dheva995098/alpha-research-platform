"""
FastAPI main application for Alpha Research Platform.
Startup, shutdown, and route initialization.
"""
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from .config import settings
from .models import init_db

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("Starting Alpha Research Platform...")
    init_db()
    logger.info("Database initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Alpha Research Platform...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Alpha Research Platform",
        description="AI-driven alpha expression research automation for WorldQuant BRAIN",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Health check endpoint
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """Return JSON for unexpected failures so the dashboard can show the real error."""
        logger.exception("Unhandled API error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)},
        )

    @app.get("/health", tags=["health"])
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "environment": settings.environment}
    
    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint."""
        return {
            "message": "Alpha Research Platform API",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
            "dashboard": "http://127.0.0.1:5173",
            "api_groups": [
                "/api/accounts",
                "/api/generation",
                "/api/orchestration",
                "/api/ml",
                "/api/filters",
            ],
        }
    
    # Import and include routers
    from .routes import accounts, filters, generation, ml, orchestration
    app.include_router(accounts.router, prefix="/api/accounts", tags=["accounts"])
    app.include_router(filters.router, prefix="/api/filters", tags=["filters"])
    app.include_router(generation.router, prefix="/api/generation", tags=["generation"])
    app.include_router(ml.router, prefix="/api/ml", tags=["ml"])
    app.include_router(orchestration.router, prefix="/api/orchestration", tags=["orchestration"])
    
    # TODO: Add phase 6 dashboard routes
    # from .routes import simulations, results, dashboard
    # app.include_router(simulations.router, prefix="/api/simulations", tags=["simulations"])
    # app.include_router(results.router, prefix="/api/results", tags=["results"])
    # app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
    
    logger.info("FastAPI application created successfully")
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower()
    )
