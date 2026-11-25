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

import pytest

from agent.conversation_loop import _log_prompt_divergence, _restore_or_build_system_prompt
from agent.surface_switch import identity_line_value
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
        assert identity_line_value(line, "Provider") == "qwen-next"

    def test_custom_prefixed_id_renders_the_name(self):
        agent = _render_agent(provider="custom:qwen-next", requested_provider="custom:qwen-next")
        assert identity_line_value(_timestamp_line(agent), "Provider") == "qwen-next"

    def test_bare_custom_route_still_renders_custom(self):
        agent = _render_agent(provider="custom", requested_provider="")
        line = _timestamp_line(agent)
        assert identity_line_value(line, "Provider") == "custom"
        # A nameless `custom:` must collapse, not survive as a provider called 'custom:'.
        assert "Provider: custom:" not in line

    def test_cloud_route_is_unchanged(self):
        agent = _render_agent(model="deepseek/deepseek-v4.1-flash-1017",
                              provider="openrouter", requested_provider="openrouter",
                              base_url="https://openrouter.ai/api/v1")
        assert identity_line_value(_timestamp_line(agent), "Provider") == "openrouter"

    def test_absent_provider_renders_no_line(self):
        """An agent with no provider must not invent one from `requested_provider`."""
        agent = _render_agent(provider="", requested_provider="stale-name")
        line = _timestamp_line(agent)
        assert identity_line_value(line, "Provider") == ""
        assert "Provider:" not in line


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


class TestProviderValueEdgeCases:
    """Spellings that are not routes: a nameless ``custom:`` and junk in ``requested_provider``."""

    @staticmethod
    def _probe(agent, stored: str) -> bool:
        from agent.conversation_loop import _stored_prompt_matches_runtime
        return _stored_prompt_matches_runtime(agent, stored)

    def test_custom_prefix_without_a_name_is_the_bare_wire(self):
        agent = _render_agent(provider="custom:", requested_provider="")
        line = _timestamp_line(agent)
        assert identity_line_value(line, "Provider") == "custom"
        assert "Provider: custom:" not in line
        assert self._probe(agent, "body\n\n" + _timestamp_line(agent))

    def test_junk_requested_provider_is_not_an_identity_on_a_named_route(self):
        """`requested_provider` only names the route when the wire id was collapsed."""
        agent = _restore_agent(_stored_prompt("5"), provider="openrouter",
                               requested_provider="5", base_url="https://openrouter.ai/api/v1")

        _restore(agent, _stored_prompt("5"))

        agent._build_system_prompt.assert_called_once()

    def test_surrounding_whitespace_does_not_create_a_second_identity(self):
        spaced = _render_agent(provider="  Qwen-Next  ", requested_provider="  qwen-next ")
        assert identity_line_value(_timestamp_line(spaced), "Provider") == "qwen-next"
        assert self._probe(spaced, "body\n\n" + _timestamp_line(_render_agent()))


class TestWriterAndProbeAgree:
    """What the builder writes into the trailer, the probe must accept.

    This is the pairing the change exists to guarantee; a test that hand-writes the ``Provider:``
    literal can pass while the writer and the probe disagree (which is how a failover-restore path
    writing the collapsed wire id escaped the first pass).
    """

    SHAPES = [
        ("gateway_named", {"provider": "custom", "requested_provider": "qwen-next"}),
        ("picker", {"provider": "custom:qwen-next", "requested_provider": "custom:qwen-next"}),
        ("bare_name", {"provider": "qwen-next", "requested_provider": "qwen-next"}),
        ("bare_custom", {"provider": "custom", "requested_provider": ""}),
        ("display_name", {"provider": "custom", "requested_provider": "Strata Mint"}),
        ("cloud", {"provider": "openrouter", "requested_provider": "openrouter"}),
        ("anthropic", {"provider": "anthropic", "requested_provider": "anthropic"}),
        ("moa", {"provider": "moa", "requested_provider": "moa"}),
    ]

    @staticmethod
    def _probe(agent, stored: str) -> bool:
        from agent.conversation_loop import _stored_prompt_matches_runtime
        return _stored_prompt_matches_runtime(agent, stored)

    @pytest.mark.parametrize("name,identity", SHAPES, ids=[s[0] for s in SHAPES])
    def test_the_trailer_the_builder_writes_is_accepted_by_the_probe(self, name, identity):
        agent = _render_agent(**identity)
        assert self._probe(agent, "body\n\n" + _timestamp_line(agent)), name

    def test_display_name_and_key_spellings_are_one_identity(self):
        """`providers.strata-mint.name: Strata Mint` must not announce two providers."""
        gateway = _render_agent(provider="custom", requested_provider="Strata Mint")
        assert identity_line_value(_timestamp_line(gateway), "Provider") == "strata-mint"

        switch = _render_agent(provider="custom:strata-mint",
                               requested_provider="custom:strata-mint")
        assert self._probe(gateway, "body\n\n" + _timestamp_line(switch))
        assert self._probe(switch, "body\n\n" + _timestamp_line(gateway))


class TestRestorePathKeepsTheLabel:
    def test_failover_restore_rewrites_the_name_not_the_wire_id(self, monkeypatch):
        """The undo-writer must restore the bytes the builder would write (#141035)."""
        from agent import route_binding

        for target in (
            "agent.agent_runtime_helpers._restore_runtime_capabilities",
            "agent.agent_runtime_helpers._rebuild_primary_client",
            "agent.route_verdicts.reset_route_verdicts",
            "agent.chat_completion_helpers._reset_stale_streak",
        ):
            monkeypatch.setattr(target, lambda *a, **k: None)

        stored = "body\n\n" + _timestamp_line(_render_agent())
        agent = MagicMock()
        agent._cached_system_prompt = stored
        rt = {
            "model": MODEL, "provider": "custom", "requested_provider": "qwen-next",
            "base_url": QWEN_URL, "api_mode": "chat_completions", "api_key": "k",
            "request_overrides": {}, "client_kwargs": {}, "use_prompt_caching": True,
            "use_native_cache_layout": False, "compressor_model": MODEL,
            "compressor_context_length": 262144, "compressor_base_url": QWEN_URL,
            "compressor_api_key": "k", "compressor_provider": "custom",
        }

        route_binding.reinstall_runtime_snapshot(agent, rt)

        assert agent._cached_system_prompt == stored


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


class TestEveryWriterUsesTheLabel:
    """Three places rewrite the cached trailer mid-turn; each must write the same label.

    An audit proved the voice and fallback writers were never executed by any test, which is
    exactly how the failover-restore regression survived the first review pass.
    """

    def test_voice_turn_writer_uses_the_label(self, monkeypatch):
        from agent import voice_turn_route as vtr

        stored = "body\n\n" + _timestamp_line(_render_agent())
        agent = MagicMock()
        agent._voice_turn_pending = True
        agent._route_pool_choice = None
        agent.reasoning_config = {}
        agent._client_kwargs = {}
        agent.request_overrides = {}
        agent.model = MODEL
        agent.provider = "custom"
        agent.requested_provider = "qwen-next"
        agent.base_url = QWEN_URL
        agent.api_mode = "chat_completions"
        agent._cached_system_prompt = stored

        import agent.auxiliary_task_config as atc
        monkeypatch.setattr(atc, "_get_auxiliary_task_config",
                            lambda task: {"provider": "custom", "model": MODEL, "base_url": QWEN_URL})
        monkeypatch.setattr(vtr, "_capture", lambda a: {"snapshot": {"model": MODEL}})
        monkeypatch.setattr(vtr, "_fits", lambda *a: True)

        def bind(a, entry, provider, model, **kw):
            a.model, a.provider, a.requested_provider = model, "custom", "qwen-next"
            return ("old", "old-provider")

        monkeypatch.setattr("agent.route_binding.bind_route_entry", bind)

        vtr.begin_voice_turn_route(agent, [{"role": "user", "content": "hi"}], stored)

        assert agent._cached_system_prompt == stored

    def test_fallback_activation_writer_uses_the_label(self, monkeypatch):
        from agent import chat_completion_helpers as cch

        stored = "body\n\n" + _timestamp_line(_render_agent())
        agent = MagicMock()
        agent._fallback_chain = [{"provider": "custom", "model": "m2"}]
        agent._fallback_index = 0
        agent._unavailable_fallback_keys = set()
        agent.model = MODEL
        agent.provider = "custom"
        agent.requested_provider = "qwen-next"
        agent.base_url = QWEN_URL
        agent._cached_system_prompt = stored

        import agent.fallback_cooldown as fc
        import agent.fallback_notice as fn
        monkeypatch.setattr(fc, "switch_deferred_by_reset", lambda *a, **k: False)
        monkeypatch.setattr(fc, "_arm_rate_limit_cooldown", lambda *a, **k: None)
        monkeypatch.setattr(cch, "_should_skip_fallback_candidate", lambda *a, **k: False)
        monkeypatch.setattr(fn, "announce_fallback_switch", lambda *a, **k: None)
        monkeypatch.setattr(cch, "_log_fallback_activated", lambda *a, **k: None)

        def bind(a, fb, provider, model, **kw):
            a.model, a.provider, a.requested_provider = model, "custom", "qwen-next"
            return ("old-model", "custom")

        monkeypatch.setattr("agent.route_binding.bind_route_entry", bind)

        assert cch.try_activate_fallback(agent) is True
        assert identity_line_value(agent._cached_system_prompt, "Provider") == "qwen-next"


class TestDiagnosticAccuracy:
    """The rebuild diagnostic must name the true position AND be wired into the rebuild."""

    @staticmethod
    def _records(caplog):
        return [r.getMessage() for r in caplog.records if "System prompt rebuild" in r.getMessage()]

    def test_divergence_reports_the_true_first_difference(self, caplog):
        stored = "body\n\nModel: a\nProvider: custom\n\nPlatform: discord"
        rebuilt = "body\n\nModel: a\nProvider: qwen-next\n\nPlatform: discord"
        expected = next(i for i, (a, b) in enumerate(zip(stored, rebuilt)) if a != b)

        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            _log_prompt_divergence(SimpleNamespace(session_id="s"), stored, rebuilt)

        assert f"char {expected}" in self._records(caplog)[0]

    def test_the_restore_path_logs_where_a_stale_rebuild_diverged(self, caplog):
        """Deleting the call site must fail: calling the helper directly is not enough."""
        stored = _stored_prompt("qwen-next")
        agent = _restore_agent(stored, prebuilt=_stored_prompt("openrouter"),
                               provider="openrouter", requested_provider="openrouter",
                               base_url="https://openrouter.ai/api/v1")

        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            _restore(agent, stored)

        agent._build_system_prompt.assert_called_once()
        assert any("diverges" in line for line in self._records(caplog)), self._records(caplog)

    def test_no_provider_agent_does_not_accept_a_stored_provider_name(self):
        """The probe side of 'an agent with no provider announces nothing'."""
        from agent.conversation_loop import _stored_prompt_matches_runtime
        agent = _render_agent(provider="", requested_provider="qwen-next")
        assert _stored_prompt_matches_runtime(agent, _stored_prompt("qwen-next")) is False

    def test_prompt_rebuild_without_a_stored_model_line_is_not_a_model_switch(self, monkeypatch):
        """A legacy row with no Model: line must not be reported as a model switch."""
        from hermes_cli.observability import shared_metrics_efficiency as eff

        causes = []
        monkeypatch.setattr(eff, "record_cache_break", lambda agent, cause: causes.append(cause))
        stored = "You are Hermes.\nProvider: qwen-next\n"
        same_route = SimpleNamespace(provider="custom", requested_provider="qwen-next", model="model-a")

        eff.record_prompt_rebuild(same_route, stored, "stale_runtime", stored + "Host: x\n")

        assert causes == ["system_prompt_rebuild"]
