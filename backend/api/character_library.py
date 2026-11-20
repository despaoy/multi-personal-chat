"""Authenticated, account-owned character editing and portable sharing."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.dependencies import get_current_user
from app.runtime import get_runtime_container
from character import kamimaho_preset
from character.library import CharacterPackage, CharacterSettings
from db.errors import RevisionConflictError
from repositories.character_library import CharacterLibraryRepository, CharacterNotFoundError, SourceMergeBlockedError
from repositories.character_sources import CharacterSourceRepository, OriginalSource

router = APIRouter(prefix="/api/character-library", tags=["character-library"])


def account(user=Depends(get_current_user)) -> str:
    identity = user.get("user_id")
    if identity is None or str(identity).strip() == "" or user.get("auth_type") in {"api_key", "managed_api_key"}:
        raise HTTPException(403, "角色库需要登录用户账号")
    return str(identity)


def repository(request: Request):
    return CharacterLibraryRepository(get_runtime_container(request.app).db)


class UpdateCharacter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=1)
    settings: CharacterSettings


class PublicationSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=1)
    corpus_id: str | None = Field(default=None, min_length=1, max_length=100)
    source_ids: list[str] = Field(default_factory=list, max_length=100)


class PublishCharacter(PublicationSelection):
    enabled: bool
    preview_id: str | None = Field(default=None, min_length=1, max_length=100)
    previous_publication_id: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode='after')
    def require_preview(self):
        if self.enabled and self.preview_id is None:
            raise ValueError('请先预览分享内容')
        return self


class CopyPublication(BaseModel):
    model_config = ConfigDict(extra="forbid")
    publication_id: str = Field(min_length=1, max_length=150)
    request_id: str = Field(min_length=1, max_length=100)


class SourceUpdateIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=1)
    source_revision: int = Field(ge=1)
    publication_id: str = Field(min_length=1, max_length=150)
    local_corpus_id: str | None = Field(min_length=1, max_length=100)


SourceChoices = dict[Annotated[str, Field(min_length=1, max_length=100, pattern=r'^[a-zA-Z0-9_-]+$')], Literal['local', 'author']]


class FinalSourceUpdatePreview(SourceUpdateIdentity):
    choices: SourceChoices = Field(max_length=100)


class ImportSourceUpdate(SourceUpdateIdentity):
    choices: SourceChoices | None = Field(default=None, max_length=100)
    preview_id: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')

    @model_validator(mode='after')
    def require_final_preview(self):
        if (self.choices is None) != (self.preview_id is None):
            raise ValueError('逐篇选择与最终预览编号必须一起提交')
        return self


class PublishOriginalSources(BaseModel):
    model_config = ConfigDict(extra='forbid')
    character_revision: int = Field(ge=1)
    sources: list[OriginalSource] = Field(min_length=1, max_length=100)


class RetrySourceIndex(BaseModel):
    model_config = ConfigDict(extra='forbid')
    attempt: int = Field(ge=1)


def original_sources(request: Request):
    return CharacterSourceRepository(get_runtime_container(request.app).db)


def source_call(operation, *args):
    try:
        return call(operation, *args)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def scoped_corpus(repo, owner, character_id, corpus_id):
    value = repo.get(owner, corpus_id)
    if value['character_id'] != character_id:
        raise CharacterNotFoundError('资料版本不属于当前角色')
    return value


@router.post('/{character_id}/sources', status_code=201)
def publish_sources(character_id: str, body: PublishOriginalSources, owner=Depends(account), repo=Depends(original_sources)):
    return source_call(repo.publish, owner, character_id, body.character_revision, [source.model_dump() for source in body.sources])


@router.get('/{character_id}/sources')
def list_sources(character_id: str, revision: int = Query(ge=1), limit: int = Query(50, ge=1, le=100),
                 offset: int = Query(0, ge=0), owner=Depends(account), repo=Depends(original_sources)):
    return {'corpora': source_call(repo.list, owner, character_id, revision, limit, offset)}


@router.get('/{character_id}/sources/{corpus_id}')
def get_sources(character_id: str, corpus_id: str, owner=Depends(account), repo=Depends(original_sources)):
    return source_call(scoped_corpus, repo, owner, character_id, corpus_id)


@router.delete('/{character_id}/sources/{corpus_id}')
def revoke_sources(character_id: str, corpus_id: str, owner=Depends(account), repo=Depends(original_sources)):
    source_call(scoped_corpus, repo, owner, character_id, corpus_id)
    source_call(repo.revoke, owner, corpus_id)
    return {'revoked': corpus_id}


@router.post('/{character_id}/sources/{corpus_id}/retry')
def retry_source_index(character_id: str, corpus_id: str, body: RetrySourceIndex,
                       owner=Depends(account), repo=Depends(original_sources)):
    source_call(scoped_corpus, repo, owner, character_id, corpus_id)
    return source_call(repo.retry_index_job, owner, corpus_id, body.attempt)


def call(operation, *args):
    try:
        return operation(*args)
    except CharacterNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except SourceMergeBlockedError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RevisionConflictError as exc:
        raise HTTPException(409, "角色已被修改，请刷新后重试") from exc


@router.get("")
def list_characters(
    limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0), owner=Depends(account), repo=Depends(repository)
):
    return {"characters": repo.list(owner, limit, offset)}


@router.post("", status_code=201)
def create_character(body: CharacterSettings, owner=Depends(account), repo=Depends(repository)):
    return repo.create(owner, body)


@router.post("/import", status_code=201)
def import_character(body: CharacterPackage, owner=Depends(account), repo=Depends(repository)):
    return repo.create(owner, body.settings)


@router.post("/presets/{preset_key}", status_code=201)
def install_preset(preset_key: str, owner=Depends(account), repo=Depends(original_sources)):
    if preset_key != kamimaho_preset.PRESET_KEY:
        raise HTTPException(404, "预设角色不存在")
    return source_call(repo.install_preset, owner, preset_key, kamimaho_preset.SETTINGS, kamimaho_preset.SOURCES)


@router.post('/import-preview')
def preview_character_import(body: CharacterPackage, owner=Depends(account)):
    """Validate portable settings without creating a character or executing file content."""
    return body


@router.get("/shared/{token}")
def shared_character(token: str, owner=Depends(account), repo=Depends(repository)):
    return call(repo.shared, token)


@router.post("/shared/{token}/copy", status_code=201)
def copy_character(token: str, body: CopyPublication, owner=Depends(account), repo=Depends(repository)):
    return call(repo.copy_shared, owner, token, body.publication_id, body.request_id)


@router.get("/{character_id}")
def get_character(
    character_id: str, revision: int | None = Query(None, ge=1), owner=Depends(account), repo=Depends(repository)
):
    return call(repo.get, owner, character_id, revision)


@router.put("/{character_id}")
def update_character(character_id: str, body: UpdateCharacter, owner=Depends(account), repo=Depends(repository)):
    return call(repo.update, owner, character_id, body.revision, body.settings)


@router.post("/{character_id}/sharing-preview")
def preview_sharing(character_id: str, body: PublicationSelection, owner=Depends(account), repo=Depends(repository)):
    return source_call(repo.publish_preview, owner, character_id, body.revision, body.corpus_id, body.source_ids)


@router.put("/{character_id}/sharing")
def publish_character(character_id: str, body: PublishCharacter, owner=Depends(account), repo=Depends(repository)):
    return source_call(repo.publish, owner, character_id, body.revision, body.enabled, body.corpus_id, body.source_ids,
                       body.preview_id, body.previous_publication_id)


@router.get("/{character_id}/export")
def export_character(character_id: str, owner=Depends(account), repo=Depends(repository)):
    character = call(repo.get, owner, character_id)
    return CharacterPackage(settings=character["settings"])


@router.get("/{character_id}/source-update")
def source_update(character_id: str, owner=Depends(account), repo=Depends(repository)):
    return call(repo.source_update, owner, character_id)


@router.post("/{character_id}/source-update/preview")
def preview_source_update(character_id: str, body: FinalSourceUpdatePreview, owner=Depends(account), repo=Depends(repository)):
    return source_call(repo.preview_source_update, owner, character_id, body.revision, body.source_revision,
                       body.publication_id, body.local_corpus_id, body.choices)


@router.post("/{character_id}/source-update")
def import_source_update(character_id: str, body: ImportSourceUpdate, owner=Depends(account), repo=Depends(repository)):
    return source_call(repo.import_source_update, owner, character_id, body.revision, body.source_revision,
                       body.publication_id, body.local_corpus_id, body.choices, body.preview_id)
