"""Thin API wrappers for async agent lifecycle operations."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sse_starlette.sse import EventSourceResponse

from app.api.deps import ActorContext, require_admin_or_agent, require_org_admin
from app.core.auth import AuthContext, get_auth_context
from app.db.session import get_session
from app.schemas.agents import (
    AgentCreate,
    AgentHeartbeat,
    AgentHeartbeatCreate,
    AgentRead,
    AgentSyncResponse,
    AgentUpdate,
)
from app.schemas.common import OkResponse
from app.schemas.pagination import DefaultLimitOffsetPage
from app.services.openclaw.provisioning_db import AgentLifecycleService, AgentUpdateOptions
from app.services.organizations import OrganizationContext

if TYPE_CHECKING:
    from fastapi_pagination.limit_offset import LimitOffsetPage
    from sqlmodel.ext.asyncio.session import AsyncSession

router = APIRouter(prefix="/agents", tags=["agents"])

BOARD_ID_QUERY = Query(default=None)
GATEWAY_ID_QUERY = Query(default=None)
SINCE_QUERY = Query(default=None)
SESSION_DEP = Depends(get_session)
ORG_ADMIN_DEP = Depends(require_org_admin)
ACTOR_DEP = Depends(require_admin_or_agent)
AUTH_DEP = Depends(get_auth_context)


@dataclass(frozen=True, slots=True)
class _AgentUpdateParams:
    force: bool
    auth: AuthContext
    ctx: OrganizationContext


def _agent_update_params(
    *,
    force: bool = False,
    auth: AuthContext = AUTH_DEP,
    ctx: OrganizationContext = ORG_ADMIN_DEP,
) -> _AgentUpdateParams:
    return _AgentUpdateParams(force=force, auth=auth, ctx=ctx)


AGENT_UPDATE_PARAMS_DEP = Depends(_agent_update_params)


@router.get("", response_model=DefaultLimitOffsetPage[AgentRead])
async def list_agents(
    board_id: UUID | None = BOARD_ID_QUERY,
    gateway_id: UUID | None = GATEWAY_ID_QUERY,
    session: AsyncSession = SESSION_DEP,
    ctx: OrganizationContext = ORG_ADMIN_DEP,
) -> LimitOffsetPage[AgentRead]:
    """List agents visible to the active organization admin."""
    service = AgentLifecycleService(session)
    return await service.list_agents(
        board_id=board_id,
        gateway_id=gateway_id,
        ctx=ctx,
    )


async def _sync_local_agents(*, session: AsyncSession, ctx: OrganizationContext) -> AgentSyncResponse:
    from sqlmodel import col, select

    from app.core.time import utcnow
    from app.models.agents import Agent
    from app.models.boards import Board
    from app.models.gateways import Gateway

    config_path = Path(os.getenv("OPENCLAW_CONFIG_PATH", "/root/.openclaw/openclaw.json"))
    if not config_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "OpenClaw config not found. Set OPENCLAW_CONFIG_PATH or mount "
                f"the config file into the backend container. looked_at={config_path}"
            ),
        )

    data = json.loads(config_path.read_text(encoding="utf-8"))
    listed_agents = data.get("agents", {}).get("list", [])
    agent_names = [str(item.get("id", "")).strip() for item in listed_agents]
    agent_names = [name for name in agent_names if name]
    if not agent_names:
        return AgentSyncResponse(discovered=0, created=0, updated=0)

    gateway_port = data.get("gateway", {}).get("port", 18789)
    gateway_token = data.get("gateway", {}).get("auth", {}).get("token")
    gateway_url = f"ws://127.0.0.1:{gateway_port}/ws"
    workspace_root = str(
        data.get("agents", {}).get("defaults", {}).get("workspace")
        or "/root/.openclaw/workspace"
    )

    gateway_name = "Local OpenClaw Gateway"
    gateway = (
        await session.exec(
            select(Gateway).where(
                col(Gateway.organization_id) == ctx.org.id,
                col(Gateway.name) == gateway_name,
            ),
        )
    ).first()

    now = utcnow()
    if gateway is None:
        gateway = Gateway(
            organization_id=ctx.org.id,
            name=gateway_name,
            url=gateway_url,
            token=gateway_token,
            workspace_root=workspace_root,
            created_at=now,
            updated_at=now,
        )
        session.add(gateway)
        await session.commit()
        await session.refresh(gateway)
    else:
        changed = False
        if gateway.url != gateway_url:
            gateway.url = gateway_url
            changed = True
        if gateway.token != gateway_token:
            gateway.token = gateway_token
            changed = True
        if gateway.workspace_root != workspace_root:
            gateway.workspace_root = workspace_root
            changed = True
        if changed:
            gateway.updated_at = now
            session.add(gateway)
            await session.commit()
            await session.refresh(gateway)

    board_name = "Local OpenClaw Board"
    board_slug = "local-openclaw-board"
    board = (
        await session.exec(
            select(Board).where(
                col(Board.organization_id) == ctx.org.id,
                col(Board.slug) == board_slug,
            ),
        )
    ).first()
    if board is None:
        board = Board(
            organization_id=ctx.org.id,
            gateway_id=gateway.id,
            name=board_name,
            slug=board_slug,
            description="Auto-created board for local OpenClaw agents",
            objective="Coordinate local OpenClaw team",
            max_agents=max(1, len(agent_names) - 1),
            created_at=now,
            updated_at=now,
        )
        session.add(board)
        await session.commit()
        await session.refresh(board)

    existing = (
        await session.exec(
            select(Agent).where(col(Agent.gateway_id) == gateway.id),
        )
    ).all()
    by_name = {agent.name: agent for agent in existing}

    created = 0
    updated = 0
    for idx, name in enumerate(agent_names):
        agent = by_name.get(name)
        is_lead = name == "main" or idx == 0
        if agent is None:
            agent = Agent(
                name=name,
                gateway_id=gateway.id,
                board_id=board.id,
                is_board_lead=is_lead,
                status="online",
                openclaw_session_id=f"agent:{name}:{name}",
                heartbeat_config={"interval_seconds": 15, "timeout_seconds": 60},
                last_seen_at=now,
                created_at=now,
                updated_at=now,
            )
            session.add(agent)
            created += 1
            continue

        changed = False
        if agent.board_id != board.id:
            agent.board_id = board.id
            changed = True
        if agent.gateway_id != gateway.id:
            agent.gateway_id = gateway.id
            changed = True
        if agent.status != "online":
            agent.status = "online"
            changed = True
        if bool(agent.is_board_lead) != bool(is_lead):
            agent.is_board_lead = is_lead
            changed = True
        desired_session = f"agent:{name}:{name}"
        if (agent.openclaw_session_id or "") != desired_session:
            agent.openclaw_session_id = desired_session
            changed = True
        if changed:
            agent.last_seen_at = now
            agent.updated_at = now
            session.add(agent)
            updated += 1

    await session.commit()
    return AgentSyncResponse(discovered=len(agent_names), created=created, updated=updated)


@router.post("/sync", response_model=AgentSyncResponse)
async def sync_local_agents(
    session: AsyncSession = SESSION_DEP,
    ctx: OrganizationContext = ORG_ADMIN_DEP,
) -> AgentSyncResponse:
    """Sync local OpenClaw agents from config into Mission Control."""
    return await _sync_local_agents(session=session, ctx=ctx)


@router.get("/stream")
async def stream_agents(
    request: Request,
    board_id: UUID | None = BOARD_ID_QUERY,
    since: str | None = SINCE_QUERY,
    session: AsyncSession = SESSION_DEP,
    ctx: OrganizationContext = ORG_ADMIN_DEP,
) -> EventSourceResponse:
    """Stream agent updates as SSE events."""
    service = AgentLifecycleService(session)
    return await service.stream_agents(
        request=request,
        board_id=board_id,
        since=since,
        ctx=ctx,
    )


@router.post("", response_model=AgentRead)
async def create_agent(
    payload: AgentCreate,
    session: AsyncSession = SESSION_DEP,
    actor: ActorContext = ACTOR_DEP,
) -> AgentRead:
    """Create and provision an agent."""
    service = AgentLifecycleService(session)
    return await service.create_agent(payload=payload, actor=actor)


@router.get("/{agent_id}", response_model=AgentRead)
async def get_agent(
    agent_id: str,
    session: AsyncSession = SESSION_DEP,
    ctx: OrganizationContext = ORG_ADMIN_DEP,
) -> AgentRead:
    """Get a single agent by id."""
    service = AgentLifecycleService(session)
    return await service.get_agent(agent_id=agent_id, ctx=ctx)


@router.patch("/{agent_id}", response_model=AgentRead)
async def update_agent(
    agent_id: str,
    payload: AgentUpdate,
    params: _AgentUpdateParams = AGENT_UPDATE_PARAMS_DEP,
    session: AsyncSession = SESSION_DEP,
) -> AgentRead:
    """Update agent metadata and optionally reprovision."""
    service = AgentLifecycleService(session)
    return await service.update_agent(
        agent_id=agent_id,
        payload=payload,
        options=AgentUpdateOptions(
            force=params.force,
            user=params.auth.user,
            context=params.ctx,
        ),
    )


@router.post("/{agent_id}/heartbeat", response_model=AgentRead)
async def heartbeat_agent(
    agent_id: str,
    payload: AgentHeartbeat,
    session: AsyncSession = SESSION_DEP,
    actor: ActorContext = ACTOR_DEP,
) -> AgentRead:
    """Record a heartbeat for a specific agent."""
    service = AgentLifecycleService(session)
    return await service.heartbeat_agent(agent_id=agent_id, payload=payload, actor=actor)


@router.post("/heartbeat", response_model=AgentRead)
async def heartbeat_or_create_agent(
    payload: AgentHeartbeatCreate,
    session: AsyncSession = SESSION_DEP,
    actor: ActorContext = ACTOR_DEP,
) -> AgentRead:
    """Heartbeat an existing agent or create/provision one if needed."""
    service = AgentLifecycleService(session)
    return await service.heartbeat_or_create_agent(payload=payload, actor=actor)


@router.delete("/{agent_id}", response_model=OkResponse)
async def delete_agent(
    agent_id: str,
    session: AsyncSession = SESSION_DEP,
    ctx: OrganizationContext = ORG_ADMIN_DEP,
) -> OkResponse:
    """Delete an agent and clean related task state."""
    service = AgentLifecycleService(session)
    return await service.delete_agent(agent_id=agent_id, ctx=ctx)
