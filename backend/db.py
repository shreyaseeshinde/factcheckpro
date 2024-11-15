from sqlmodel import SQLModel, Session, create_engine
from fastapi import Depends
from typing import Annotated
from dotenv import load_dotenv
import os
# Load environment variables from the .env file
load_dotenv()


# Create a database engine
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SQLModel.metadata.create_all(engine)


# Create a session
def get_session():
    with Session(engine) as session:
        yield session


# Create a session dependency
SessionDep = Annotated[Session, Depends(get_session)]
