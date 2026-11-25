"""Regression tests for provider-alias drift in the system prompt's runtime identity.

One route, two names.  ``hermes_cli.runtime_provider_custom._custom_runtime`` collapses a
named ``providers.<name>`` entry to ``provider="custom"`` and keeps the configured name in
``requested_provider`` — that is what the gateway passes to ``AIAgent`` on every turn.  A
``/model`` switch instead carries the route as ``custom:<name>``
(``hermes_cli/providers.custom_provider_slug``) or, for a ``model_aliases`` entry, as the bare
configured name.  All of them land on the same ``base_url``.

The prompt's ``Provider:`` trailer is rendered from the agent and the restore probe compares it
against the agent, so a route with two identities classified the stored prompt
``stale_runtime``, rebuilt it, and because the system prompt is serialized ahead of the
conversation every token behind it re-prefilled (upstream #141035).

The reuse cases and the rebuild cases are paired on purpose, and the rebuild cases keep the
``Model:`` line IDENTICAL so the provider comparison is the only thing that can trigger them.
A rebuild case that also changes the model proves nothing about this change — it would pass
against a comparison that accepts every value, which is exactly the bug a reviewer found in the
first cut of this fix.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.conversation_loop import _log_prompt_divergence, _restore_or_build_system_prompt
from agent.system_prompt import _timestamp_line

QWEN_URL = "http://localhost:8005/v1"
MODEL = "qwen3.8-flash-next"


def _render_agent(**overrides) -> SimpleNamespace:
    """Minimal agent for the identity trailer only."""
    agent = SimpleNamespace(
        session_id="20261125_105109_1f1d89db",
        model=MODEL,
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
    agent.model = MODEL
    agent.provider = "custom"
    agent.requested_provider = "qwen-next"
    agent.base_url = QWEN_URL
    agent._platform_hint_overrides = None
    agent._surface_switch_note = ""
    agent._gateway_turn_context_notes = ""
    for key, value in identity.items():
        setattr(agent, key, value)
    return agent


def _stored_prompt(provider: str, model: str = MODEL) -> str:
    return (
        "You are Jane.\n\nConversation started: Tuesday, September 29, 2026 (CET, UTC+01:00)\n"
        f"Model: {model}\nProvider: {provider}\nPlatform: discord"
    )


def _restore(agent, stored: str) -> None:
    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])


class TestIdentityTrailerNamesTheRoute:
    def test_named_custom_route_renders_its_configured_name(self):
        """The gateway's agent is 'custom' on the wire but 'qwen-next' to the user."""
        line = _timestamp_line(_render_agent())
        assert "Provider: qwen-next" in line

    def test_custom_prefixed_id_renders_the_name(self):
        agent = _render_agent(provider="custom:qwen-next", requested_provider="custom:qwen-next")
        assert "Provider: qwen-next" in _timestamp_line(agent)

    def test_bare_custom_route_still_renders_custom(self):
        agent = _render_agent(provider="custom", requested_provider="")
        assert "Provider: custom" in _timestamp_line(agent)

    def test_cloud_route_is_unchanged(self):
        agent = _render_agent(model="deepseek/deepseek-v4.1-flash-1017",
                              provider="openrouter", requested_provider="openrouter",
                              base_url="https://openrouter.ai/api/v1")
        assert "Provider: openrouter" in _timestamp_line(agent)

    def test_absent_provider_renders_no_line(self):
        """An agent with no provider must not invent one from `requested_provider`."""
        agent = _render_agent(provider="", requested_provider="stale-name")
        assert "Provider:" not in _timestamp_line(agent)


class TestSameRouteAcrossSurfacesReusesThePrompt:
    def test_gateway_agent_reuses_a_prompt_written_by_a_switch(self, caplog):
        """Stored by /model as 'qwen-next' -> resumed by the gateway (provider='custom')."""
        stored = _stored_prompt("qwen-next")
        agent = _restore_agent(stored)

        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            _restore(agent, stored)

        assert agent._cached_system_prompt == stored
        agent._build_system_prompt.assert_not_called()
        assert not [r for r in caplog.records if "stale runtime identity" in r.getMessage()]

    def test_switch_path_reuses_a_gateway_written_prompt(self):
        """Stored by the gateway as 'custom' -> resumed by a `custom:<name>` switch."""
        stored = _stored_prompt("custom")
        agent = _restore_agent(stored, provider="custom:qwen-next",
                               requested_provider="custom:qwen-next")

        _restore(agent, stored)

        assert agent._cached_system_prompt == stored
        agent._build_system_prompt.assert_not_called()

    def test_bare_name_switch_rebuilds_once_then_the_gateway_reuses(self):
        """A `model_aliases` switch carries the bare name, which is not the wire id.

        That costs one rebuild at the switch (documented, bounded); the rebuild writes the name,
        so every later gateway turn replays those bytes.
        """
        switch_agent = _restore_agent(_stored_prompt("custom"), prebuilt=_stored_prompt("qwen-next"),
                                      provider="qwen-next", requested_provider="qwen-next")
        _restore(switch_agent, _stored_prompt("custom"))
        switch_agent._build_system_prompt.assert_called_once()

        rebuilt = switch_agent._cached_system_prompt
        gateway_agent = _restore_agent(rebuilt)
        _restore(gateway_agent, rebuilt)
        assert gateway_agent._cached_system_prompt == rebuilt
        gateway_agent._build_system_prompt.assert_not_called()


class TestRealRouteChangesStillRebuild:
    """A genuine switch must keep breaking the prefix: the trailer would otherwise lie.

    Model is identical on both sides in every case below, so only the provider comparison can
    force the rebuild.
    """

    def test_stored_custom_is_not_accepted_by_a_cloud_agent(self):
        """The wildcard case: 'custom' must not be an alias for OpenRouter/Anthropic/MoA."""
        agent = _restore_agent(_stored_prompt("custom"), provider="openrouter",
                               requested_provider="openrouter",
                               base_url="https://openrouter.ai/api/v1")

        _restore(agent, _stored_prompt("custom"))

        agent._build_system_prompt.assert_called_once()

    def test_stored_custom_is_not_accepted_by_an_agent_with_no_provider(self):
        agent = _restore_agent(_stored_prompt("custom"), provider="", requested_provider="")

        _restore(agent, _stored_prompt("custom"))

        agent._build_system_prompt.assert_called_once()

    def test_one_named_custom_route_to_another_rebuilds(self):
        """strata-mint and qwen-next serve the same model but are different endpoints."""
        agent = _restore_agent(_stored_prompt("strata-mint"), provider="custom:qwen-next",
                               requested_provider="custom:qwen-next")

        _restore(agent, _stored_prompt("strata-mint"))

        agent._build_system_prompt.assert_called_once()

    def test_custom_to_cloud_rebuilds(self):
        agent = _restore_agent(_stored_prompt("custom"), provider="anthropic",
                               requested_provider="anthropic",
                               base_url="https://api.anthropic.com")

        _restore(agent, _stored_prompt("custom"))

        agent._build_system_prompt.assert_called_once()

    def test_same_route_with_a_new_model_still_rebuilds(self):
        stored = _stored_prompt("qwen-next", model="qwen3.8-flash-next-old")
        agent = _restore_agent(stored)

        _restore(agent, stored)

        agent._build_system_prompt.assert_called_once()


class TestRebuildDivergenceDiagnostic:
    """A rebuild of a continuing session must say WHERE it diverged."""

    @staticmethod
    def _records(caplog):
        return [r.getMessage() for r in caplog.records if "System prompt rebuild" in r.getMessage()]

    def test_reports_the_enclosing_heading(self, caplog):
        stored = "body\n\n# Hermes runtime environment\nHost: a\n\n## Skills\n- one\n"
        rebuilt = "body\n\n# Hermes runtime environment\nHost: a\n\n## Skills\n- two\n"
        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            _log_prompt_divergence(SimpleNamespace(session_id="s"), stored, rebuilt)

        line = self._records(caplog)[0]
        assert "diverges" in line and "'## Skills'" in line

    def test_reports_a_truncated_rebuild_as_a_prefix(self, caplog):
        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            _log_prompt_divergence(SimpleNamespace(session_id="s"), "abcdef", "abc")

        line = self._records(caplog)[0]
        assert "prefix" in line and "diverges" not in line

    def test_identical_bytes_log_nothing(self, caplog):
        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            _log_prompt_divergence(SimpleNamespace(session_id="s"), "abc", "abc")

        assert self._records(caplog) == []

    def test_non_string_inputs_are_a_noop(self, caplog):
        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            _log_prompt_divergence(SimpleNamespace(session_id="s"), None, "abc")

        assert self._records(caplog) == []
