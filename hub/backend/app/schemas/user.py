from typing import Optional
from uuid import UUID
from pydantic import BaseModel, EmailStr

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str
    # Required while the Hub's registration mode is ``invite`` (stage 4.3).
    invite_token: Optional[str] = None

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    two_factor_enabled: Optional[bool] = None

class UserOut(UserBase):
    id: UUID
    is_active: bool
    two_factor_enabled: bool
    legion_account_id: Optional[UUID] = None
    is_owner: bool = False

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenPayload(BaseModel):
    sub: Optional[str] = None
