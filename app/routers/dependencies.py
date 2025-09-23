from fastapi import Header, HTTPException, status
from typing import Optional

from ..core.config import settings
from ..core.security import verify_token

async def get_token_header(x_token: str = Header(...)):
    """Verify API token"""
    if not verify_token(x_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

