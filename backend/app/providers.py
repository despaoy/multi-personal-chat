"""FastAPI providers for application-level dependencies."""

from fastapi import Request

from app.runtime import get_runtime_container
from repositories.messages import DatabaseMessageRepository, MessageRepository
from repositories.user_data import DatabaseUserDataRepository, UserDataRepository
from services.model_management import ModelManagementService, ModelManagerService


def get_message_repository(request: Request) -> MessageRepository:
    """Resolve a message repository from the current application's container."""

    container = get_runtime_container(request.app)
    return DatabaseMessageRepository(container.db)


def get_user_data_repository(request: Request) -> UserDataRepository:
    """Resolve a user-data repository from the current application's container."""

    container = get_runtime_container(request.app)
    return DatabaseUserDataRepository(container.db)


def get_model_management_service() -> ModelManagementService:
    """Resolve the concrete model manager at the application's composition edge."""

    from inference.model_manager import get_model_manager

    return ModelManagerService(get_model_manager())
