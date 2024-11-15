from sqlmodel import Session, select, update
from models import User


# User CRUD operations
class UserCRUD:

    @staticmethod
    def create(session: Session, user: User) -> User:
        user = User.model_validate(user)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

    @staticmethod
    def get(session: Session) -> list[User]:
        statement = select(User)
        users = session.exec(statement).all()
        return users

    @staticmethod
    def retrieve(session: Session, username: str) -> User | None:
        statement = select(User).where(User.username == username)
        user = session.exec(statement).first()
        return user

    @staticmethod
    def update(session: Session, user: User) -> User | None:
        statement = update(User).where(User.id == User.id).values(**user.model_dump())
        session.exec(statement)
        session.commit()
        session.refresh(user)
        return user
