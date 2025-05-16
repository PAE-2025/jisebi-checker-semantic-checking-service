import httpx
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class AuthService:
    def __init__(self, settings):
        self.auth_service_url = settings.AUTH_SERVICE_URL

    async def validate_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validates a token with the authentication service
        Returns: (is_valid, user_info)
        """
        try:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": token}
                response = await client.get(
                    self.auth_service_url,
                    headers=headers,
                    timeout=10.0  # 10 second timeout
                )
                
                # Log the response for debugging
                logger.debug(f"Auth service response: {response.status_code} - {response.text}")
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status", False):
                        return True, data.get("data", {})
                    return False, None
                return False, None
        except Exception as e:
            logger.error(f"Error validating token: {str(e)}")
            return False, None