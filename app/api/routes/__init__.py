from . import documents, query

ROUTE_MODULES = {
    "documents": documents,
    "query": query,
}

ROUTES_CONFIG = [
    {
        "router": documents.router,
        "prefix": "/documents",
        "tags": ["documents"],
        "dependencies": [],
    },
    {
        "router": query.router, 
        "prefix": "/query",
        "tags": ["query"],
        "dependencies": [],
    },
]

__all__ = [
    "documents",
    "query",
    "ROUTE_MODULES",
    "ROUTES_CONFIG",
]