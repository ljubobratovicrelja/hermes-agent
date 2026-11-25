"""Regression tests for provider-alias drift in the system prompt's runtime identity.

One route, two names.  ``hermes_cli.runtime_provider_custom._custom_runtime`` collapses a
named ``providers.<name>`` entry to ``provider="custom"`` and keeps the configured name in
``requested_provider`` — that is what the gateway passes to ``AIAgent`` on every turn.  A
``/model`` switch instead constructs the agent with the configured name as ``provider``
(``gateway/session_cli_turn.py``).  Both land on the same ``base_url``.

The prompt's ``Provider:`` trailer is rendered from ``agent.provider``
(``agent/system_prompt.py::_timestamp_line``) and the restore probe compares it against
``agent.provider`` (``agent/conversation_loop.py::_stored_prompt_matches_runtime``).  So the
two surfaces disagree about the identity of an unchanged route, ``stored_state`` becomes
``stale_runtime``, the prompt is rebuilt, and because the system prompt is serialized ahead
of the conversation every token behind it re-prefills (observed: a ~40k-token prefill per
turn, prompt-cache hit collapsing to a constant ~16.7k).  Upstream #141035.

The tests are paired on purpose: the reuse cases must stop rebuilding, and the switch cases
must keep rebuilding.  A fix that merely makes the comparison permissive fails the second
group.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.conversation_loop import _restore_or_build_system_prompt
from agent.system_prompt import _timestamp_line

QWEN_URL = "http://localhost:8005/v1"
STRATA_URL = "http://100.120.249.124:8005/v1"


def _render_agent(**overrides) -> SimpleNamespace:
    """Minimal agent for the identity trailer only."""
    agent = SimpleNamespace(
        session_id="20261125_105109_1f1d89db",
        model="qwen3.8-flash-next",
        provider="custom",
        requested_provider="qwen-next",
        platform="discord",
        pass_session_id=False,
        _session_db=None,
        _bot_chat_timeless_prompt=False,
        base_url=QWEN_URL,
    )
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


def _restore_agent(stored: str, *, prebuilt: str = "REBUILT", **identity) -> MagicMock:
    """MagicMock agent shaped like the ones ``_restore_or_build_system_prompt`` builds."""
    from tests.agent.test_system_prompt_restore import _make_agent

    db = MagicMock()
    db.get_session.return_value = {"system_prompt": stored}
    agent = _make_agent(session_db=db, prebuilt_prompt=prebuilt)
    agent.model = "qwen3.8-flash-next"
    agent.provider = "custom"
    agent.requested_provider = "qwen-next"
    agent.base_url = QWEN_URL
    agent._platform_hint_overrides = None
    agent._surface_switch_note = ""
    agent._gateway_turn_context_notes = ""
    for key, value in identity.items():
        setattr(agent, key, value)
    return agent


def _stored_prompt(provider: str, model: str = "qwen3.8-flash-next") -> str:
    return (
        "You are Jane.\n\nConversation started: Tuesday, September 29, 2026 (CET, UTC+01:00)\n"
        f"Model: {model}\nProvider: {provider}\nPlatform: discord"
    )


class TestIdentityTrailerNamesTheRoute:
    def test_named_custom_route_renders_its_configured_name(self):
        """A named custom provider is 'qwen-next' to the user and to config, not 'custom'."""
        line = _timestamp_line(_render_agent())
        assert "Provider: qwen-next" in line

    def test_bare_custom_route_still_renders_custom(self):
        agent = _render_agent(provider="custom", requested_provider="")
        assert "Provider: custom" in _timestamp_line(agent)

    def test_cloud_route_is_unchanged(self):
        agent = _render_agent(model="deepseek/deepseek-v4.1-flash-1017",
                              provider="openrouter", requested_provider="openrouter",
                              base_url="https://openrouter.ai/api/v1")
        assert "Provider: openrouter" in _timestamp_line(agent)


class TestSameRouteAcrossSurfacesReusesThePrompt:
    """The two ways to reach one base_url must not look like a provider change."""

    def test_gateway_agent_reuses_a_prompt_written_by_model_switch(self, caplog):
        """Stored by /model (Provider: qwen-next) -> resumed by the gateway (provider=custom)."""
        stored = _stored_prompt("qwen-next")
        agent = _restore_agent(stored)

        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(
                agent, None, [{"role": "user", "content": "hi"}])

        assert agent._cached_system_prompt == stored
        agent._build_system_prompt.assert_not_called()
        assert not [r for r in caplog.records if "stale runtime identity" in r.getMessage()]

    def test_model_switch_reuses_a_prompt_written_by_the_gateway(self, caplog):
        """Stored by the gateway (Provider: custom) -> resumed by /model (provider=qwen-next)."""
        stored = _stored_prompt("custom")
        agent = _restore_agent(stored, provider="qwen-next", requested_provider="qwen-next")

        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            _restore_or_build_system_prompt(
                agent, None, [{"role": "user", "content": "hi"}])

        assert agent._cached_system_prompt == stored
        agent._build_system_prompt.assert_not_called()


class TestRealRouteChangesStillRebuild:
    """A genuine switch must keep breaking the prefix: the trailer would otherwise lie."""

    def test_cloud_to_local_rebuilds(self):
        stored = _stored_prompt("openrouter", model="deepseek/deepseek-v4.1-flash-1017")
        agent = _restore_agent(stored)

        _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        agent._build_system_prompt.assert_called_once()
        assert agent._cached_system_prompt == "REBUILT"

    def test_one_named_custom_route_to_another_rebuilds(self):
        """strata-mint and qwen-next share a model family but are different endpoints."""
        stored = _stored_prompt("strata-mint", model="qwen3.8-flash-next-iq2_xs")
        agent = _restore_agent(stored, model="qwen3.8-flash-next")

        _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        agent._build_system_prompt.assert_called_once()

    def test_same_route_with_a_new_model_still_rebuilds(self):
        stored = _stored_prompt("qwen-next", model="qwen3.8-flash-next-old")
        agent = _restore_agent(stored)

        _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

        agent._build_system_prompt.assert_called_once()
