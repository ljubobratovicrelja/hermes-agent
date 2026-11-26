"""Per-channel working directory for messaging gateway sessions.

``ChannelOverride.cwd`` pins where a chat's turns run: the terminal tool's per-session
cwd record is seeded from it on a fresh chat, and the logical cwd (project-root
discovery, project skills, AGENTS.md/context files) is bound for the agent run. The
tests drive ``TurnRunner.run_sync`` with the deep scoped run stubbed and capture the
live cwd state from inside it, so a missing call site — not just a missing helper —
dies. ``_make_runner`` follows ``test_internal_event_pin_wiring``'s fixture convention.

Loader half: ``cwd`` must survive every YAML→``ChannelOverride`` conversion, including
the multiplexed per-profile re-read (``_profile_channel_overrides``) — a class that
gained the field while the loaders dropped it would be dead code.
"""

from __future__ import annotations

import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.runtime_cwd import scoped_session_cwd
from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionEntry, SessionSource
from tools.terminal_tool import clear_session_cwd, get_session_cwd, record_session_cwd

KEY = "agent:main:discord:group:420000000000000001:420000000000000003"
THREAD_KEY = "agent:main:discord:thread:420000000000000002:420000000000000003"
CHANNEL_ID = "420000000000000001"
THREAD_ID = "420000000000000002"
PARENT_ID = "420000000000000001"

_ORIGIN = dict(
    platform=Platform.DISCORD,
    chat_id=CHANNEL_ID,
    chat_type="group",
    user_id="420000000000000003",
    scope_id="420000000000000004",
)


def _human_source(chat_id: str = CHANNEL_ID, **extra) -> SessionSource:
    return SessionSource(
        **{**_ORIGIN, "chat_id": chat_id},
        **extra,
        chat_name="Guild / #general",
        user_name="Ace",
        message_id="420000000000000005",
    )


def _thread_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=THREAD_ID,
        chat_type="thread",
        thread_id=THREAD_ID,
        user_id="420000000000000003",
        scope_id="420000000000000004",
        parent_chat_id=PARENT_ID,
        chat_name="Guild / #dev / build thread",
        user_name="Ace",
        message_id="420000000000000006",
    )


def _make_runner(monkeypatch, config: GatewayConfig | None = None, *, key: str = KEY):
    """Same fixture convention as ``test_internal_event_pin_wiring._make_runner``."""
    import agent.model_metadata as mm
    import gateway.run as gr
    import gateway.session as gs

    monkeypatch.setattr(gs, "_discord_tools_loaded", lambda: True)
    monkeypatch.setattr(gr, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr(mm, "get_model_context_length", lambda *a, **k: 100_000)

    r = gr.GatewayRunner(config or GatewayConfig())
    r.adapters = {}
    r._running_agents = {}
    r._pending_messages = {}
    r._is_user_authorized = lambda s: True
    r._set_session_env = lambda c: None
    r._handle_active_session_busy_message = AsyncMock(return_value=False)
    r._session_db = MagicMock()
    r._recover_telegram_topic_thread_id = lambda s: None
    r._cache_session_source = lambda k, s: None
    r._is_session_run_current = lambda k, g: True
    r._begin_session_run_generation = lambda k: 1
    r._reply_anchor_for_event = lambda e: None
    r._get_guild_id = lambda e: None
    r._should_send_voice_reply = lambda *a, **k: False
    r.hooks = MagicMock()
    r.hooks.emit = AsyncMock()
    r._turn_leases = None

    store = MagicMock()
    store.get_or_create_session.return_value = SessionEntry(
        session_key=key,
        session_id="sess-cwd",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 2),
        platform=Platform.DISCORD,
        chat_type="group",
    )
    store.load_transcript.return_value = []
    store.has_platform_message_id.return_value = False
    r.session_store = store
    return r


def _run_turn(runner, source, key, seen: list):
    """Drive ``TurnRunner.run_sync`` for one turn with the deep scoped run stubbed.

    The seed/bind lives in ``run_sync`` (``_run_sync_scoped`` is where the agent loop
    would run), so the capture happens between the seed and the finally-reset: a missing
    call site or a leaked token dies.
    """
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    ctx = TurnContext(
        source=source, context_prompt="ctx", channel_prompt=None,
        session_key=key, session_id="sess-cwd", run_generation=1, history=[],
    )
    tr = TurnRunner(runner, ctx)

    def fake_scoped():
        seen.append({"recorded": get_session_cwd(key), "logical": scoped_session_cwd()})
        return {"final_response": "ok", "messages": [], "api_calls": 0}

    tr._run_sync_scoped = fake_scoped
    return tr.run_sync()


def _config_with_cwd(cwd: str | None) -> GatewayConfig:
    config = GatewayConfig()
    override = ChannelOverride(cwd=cwd) if cwd is not None else ChannelOverride(model="x/m")
    config.platforms[Platform.DISCORD] = PlatformConfig(
        enabled=True, channel_overrides={CHANNEL_ID: override},
    )
    return config


@pytest.fixture(autouse=True)
def _clean_session_cwd():
    # The cwd record dict is process-local; this file shares one pytest process.
    for key in (KEY, THREAD_KEY):
        clear_session_cwd(key)
    yield
    for key in (KEY, THREAD_KEY):
        clear_session_cwd(key)


# ---------------------------------------------------------------------------
# Per-turn application (run_sync call site: seed, logical binding, finally-reset).
# ---------------------------------------------------------------------------

def test_override_cwd_seeds_fresh_session_and_binds_logical_cwd(monkeypatch, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    runner = _make_runner(monkeypatch, _config_with_cwd(str(proj)))
    seen: list = []

    result = _run_turn(runner, _human_source(), KEY, seen)

    assert result["final_response"] == "ok"
    assert seen[0]["recorded"] == str(proj), "override cwd did not seed the session cwd record"
    assert seen[0]["logical"] == str(proj), "the run's logical cwd was not bound to the override"
    # The seed persists after the turn (sticky until /new); the token reset behind it.
    assert get_session_cwd(KEY) == str(proj)
    assert scoped_session_cwd() == ""


def test_in_chat_cd_record_wins_over_override(monkeypatch, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    sticky = tmp_path / "cd-wins"
    sticky.mkdir()
    runner = _make_runner(monkeypatch, _config_with_cwd(str(proj)))
    seen: list = []

    record_session_cwd(KEY, str(sticky))
    _run_turn(runner, _human_source(), KEY, seen)

    assert seen[0]["recorded"] == str(sticky), "the override overwrote the in-chat sticky cwd"
    assert get_session_cwd(KEY) == str(sticky)


def test_thread_child_inherits_parent_channel_cwd(monkeypatch, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    config = GatewayConfig()
    config.platforms[Platform.DISCORD] = PlatformConfig(
        enabled=True, channel_overrides={PARENT_ID: ChannelOverride(cwd=str(proj))},
    )
    runner = _make_runner(monkeypatch, config, key=THREAD_KEY)
    seen: list = []

    _run_turn(runner, _thread_source(), THREAD_KEY, seen)

    assert seen[0]["recorded"] == str(proj), "thread did not inherit the parent channel's cwd"
    assert seen[0]["logical"] == str(proj)


def test_no_override_leaves_cwd_untouched(monkeypatch):
    runner = _make_runner(monkeypatch, GatewayConfig())
    seen: list = []

    result = _run_turn(runner, _human_source(), KEY, seen)

    assert result["final_response"] == "ok"
    assert seen[0]["recorded"] is None, "a chat without an override got a seeded cwd"
    assert seen[0]["logical"] == ""
    assert get_session_cwd(KEY) is None


def test_missing_cwd_directory_warns_and_falls_back(monkeypatch, tmp_path, caplog):
    missing = str(tmp_path / "does-not-exist")
    runner = _make_runner(monkeypatch, _config_with_cwd(missing))
    seen: list = []

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        result = _run_turn(runner, _human_source(), KEY, seen)

    assert result["final_response"] == "ok", "a bad cwd config must never fail the turn"
    assert seen[0]["recorded"] is None, "a missing directory was seeded anyway"
    assert get_session_cwd(KEY) is None
    assert "does-not-exist" in caplog.text, "no warning naming the configured value"


def test_tilde_cwd_expands_to_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    proj = home / "dev" / "acme-analytics"
    proj.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    runner = _make_runner(monkeypatch, _config_with_cwd("~/dev/acme-analytics"))
    seen: list = []

    _run_turn(runner, _human_source(), KEY, seen)

    assert seen[0]["recorded"] == str(proj), "~ was not expanded to the real home directory"
    assert seen[0]["logical"] == str(proj)


# ---------------------------------------------------------------------------
# Loader half: the key must survive every YAML→ChannelOverride conversion.
# A field added to the dataclass but dropped by a loader would be dead code.
# ---------------------------------------------------------------------------

def test_platform_config_from_dict_keeps_cwd():
    pc = PlatformConfig.from_dict({
        "enabled": True,
        "channel_overrides": {CHANNEL_ID: {"cwd": "~/dev/acme-analytics", "model": "a/b"}},
    })
    assert pc.channel_overrides[CHANNEL_ID].cwd == "~/dev/acme-analytics"
    # to_dict filters None; the dump rides back through from_dict unchanged.
    dumped = pc.to_dict()["channel_overrides"][CHANNEL_ID]
    assert dumped["cwd"] == "~/dev/acme-analytics"
    assert PlatformConfig.from_dict({"channel_overrides": {CHANNEL_ID: dumped}}) \
        .channel_overrides[CHANNEL_ID].cwd == "~/dev/acme-analytics"


def test_channel_override_to_dict_omits_unset_cwd():
    assert "cwd" not in ChannelOverride(model="a/b").to_dict()
    assert ChannelOverride(cwd="~/x").to_dict() == {"cwd": "~/x"}


def test_get_channel_override_returns_cwd_with_parent_precedence():
    from gateway.run import _get_channel_override

    config = GatewayConfig()
    config.platforms[Platform.DISCORD] = PlatformConfig(
        enabled=True, channel_overrides={PARENT_ID: ChannelOverride(cwd="~/dev/acme-analytics")},
    )
    ov = _get_channel_override(
        config, Platform.DISCORD, THREAD_ID, thread_id=THREAD_ID, parent_id=PARENT_ID,
    )
    assert ov is not None and ov.cwd == "~/dev/acme-analytics"


def test_profile_channel_overrides_loader_keeps_cwd(monkeypatch, tmp_path):
    """The multiplexed re-read rebuilds a GatewayConfig from the raw YAML dict; the key
    must survive that path too or routed profiles silently lose the binding."""
    import gateway.run as gr
    from gateway.run_config_loaders import _profile_channel_overrides

    raw = {"platforms": {"telegram": {
        "enabled": True,
        "channel_overrides": {"888": {"cwd": "~/dev/acme-analytics", "model": "a/b"}},
    }}}
    monkeypatch.setattr(gr, "_load_gateway_config", lambda path=None: raw)
    loaded = _profile_channel_overrides(tmp_path, Platform.TELEGRAM)
    assert loaded.platforms[Platform.TELEGRAM].channel_overrides["888"].cwd == "~/dev/acme-analytics"
