from sqlmodel import SQLModel, Field


# Shared properties


# Database model, database table inferred from class name
class User(SQLModel, table=True):
    __tablename__ = "users"
    id: int | None = Field(default=None, primary_key=True)
    name: str
    username: str
    password: str
