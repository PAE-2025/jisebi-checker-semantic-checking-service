from fastapi import Request, Depends, HTTPException
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
from typing import List
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from src.core.requests.authentication_service import AuthService
from src.config import settings

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

class AuthenticationMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, 
        app: ASGIApp, 
        auth_service: AuthService,
        exclude_paths: List[str] = None
    ):
        super().__init__(app)
        self.auth_service = auth_service
        self.exclude_paths = exclude_paths or ["/docs"]
        self.audience = settings.SELF_URL
    
    async def dispatch(self, request: Request, call_next):
        try:
            # Skip authentication for excluded paths
            if any(request.url.path.startswith(path) for path in self.exclude_paths):
                return await call_next(request)
            
            if request.url.path.startswith("/internal"):
                # Optionally verify Google OIDC token
                auth_header = request.headers.get("Authorization")
                if not auth_header or not auth_header.startswith("Bearer "):
                    raise HTTPException(status_code=401, detail="Missing or invalid authorization header for internal task")

                token = auth_header.split(" ")[1]

                try:
                    id_info = id_token.verify_oauth2_token(
                        token,
                        google_requests.Request(),
                        self.audience
                    )
                except Exception as e:
                    logger.error(f"OIDC token verification failed: {e}")
                    raise HTTPException(status_code=403, detail="Invalid internal task identity")

                # All good, continue processing
                return await call_next(request)
            
            # Extract token from the request
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                raise HTTPException(
                    status_code=401,
                    detail= {
                        "status": False,
                        "message":"Authorization header missing" 
                    },
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            # Validate with auth service
            try:
                is_valid, user_info = await self.auth_service.validate_token(auth_header)

                logger.info(f"Authentication result: valid={is_valid}")
                
                if not is_valid:
                    raise HTTPException(
                        status_code=401,
                        detail= {
                            "status": False,
                            "message": "Invalid or expired token"
                        },
                        headers={"WWW-Authenticate": "Bearer"}
                    )
                
                # Attach user info to request state for later use in route handlers
                request.state.user = user_info
                
            except HTTPException as e:
                # Re-raise HTTP exceptions
                raise
            except Exception as e:
                logger.error(f"Authentication error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "status": False,
                        "message": "Authentication service error"
                    }
                )
        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content=e.detail)
        
        # Continue processing the request
        return await call_next(request)