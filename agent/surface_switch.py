"""Surface switch without a prompt rebuild (#104414).

A surface switch (desktop <-> TUI, a session resumed under a different host) changes only which
interface renders the reply, but the surface guidance and the ``Platform:`` trailer are embedded
in the persisted system prompt.  Rebuilding for it diverged the prompt within its first blocks,
so the ENTIRE request behind it re-prefilled — a 220K-token session came back at a 1% cache hit.
The stored bytes are therefore kept and the CURRENT surface's guidance is delivered on the
per-turn user-message channel instead: that lands after the cached prefix and is stamped into
the byte-stable ``api_content`` sidecar so later turns replay it unchanged.  The prompt itself
converges at the next rebuild boundary (compaction).
"""
from __future__ import annotations

import logging
from typing import Any, List

from session_rules.turns import flatten_message_text
from agent.prompt_builder import RUNTIME_ENVIRONMENT_END, RUNTIME_ENVIRONMENT_HEADING

logger = logging.getLogger("run_agent")

_SURFACE_SWITCH_NOTE_PREFIX = "[System: This conversation is now being answered on a different interface: "
# Closes the surface name in the note; platform names are free-form for plugin platforms, so the
# terminator (not ".") delimits the parse.
_SURFACE_NAME_END = " — any earlier interface guidance"
# Only the newest note matters, and the note is re-stamped on the switch turn, so a bounded tail
# scan is enough; without a bound every turn of a never-switched session walks the whole transcript.
_NOTE_SCAN_TAIL = 200


def split_runtime_boundary(prompt: str) -> tuple:
    """``(identity, runtime_marker, runtime)`` of a persisted prompt.  Legacy prose may quote
    the runtime heading, but only the new renderer ENDS in the boundary; when the marker is
    empty the whole prompt is identity."""
    identity, runtime_marker, runtime = prompt.rpartition(f"\n\n{RUNTIME_ENVIRONMENT_HEADING}\n\n")
    return (identity, runtime_marker, runtime) if prompt.endswith(RUNTIME_ENVIRONMENT_END) else (prompt, "", "")


def runtime_host_value(prompt: str, label: str) -> str:
    """``Label: value`` from the host lines of a persisted prompt (``Current working directory``);
    new prompts delimit runtime hints, legacy prompts put them before context.  "" when absent."""
    _identity, runtime_marker, runtime = split_runtime_boundary(prompt)
    prefix = f"{label}:"
    host_lines = (runtime.split("\n\n", 1)[0] if runtime_marker else prompt).splitlines()
    for idx, line in enumerate(host_lines):
        if line.startswith("User home directory:"):
            for candidate in host_lines[idx + 1: idx + 4]:
                if candidate.startswith(prefix):
                    return candidate[len(prefix):].strip()
    return ""


def identity_line_value(prompt: str, label: str) -> str:
    """``Label: value`` from the trailer paragraph that ends the identity portion (the final
    runtime block is embedder prose, never identity).  Memory, context files and plugin prose
    come earlier and may hold ``Label:`` lines of their own; they are never read as identity,
    even when the trailer omits the label because its value is empty."""
    prefix = f"{label}:"
    trailer = split_runtime_boundary(prompt)[0].rsplit("\n\n", 1)[-1]
    matches = [line[len(prefix):].strip() for line in trailer.splitlines() if line.startswith(prefix)]
    return matches[-1] if matches else ""


# Provider ids that name no route of their own: the wire provider behind a configured
# ``providers.<name>`` entry.  A trailer that says one of these says nothing the user chose.
_COLLAPSED_PROVIDER_LABELS = frozenset({"custom", "auto", "main"})


def runtime_provider_label(agent: Any) -> str:
    """The provider name a prompt announces and the restore probe compares (#141035).

    A configured named provider resolves to the anonymous wire provider ``custom`` with the
    configured name kept in ``requested_provider`` (``hermes_cli.runtime_provider_custom``),
    a ``/model`` switch may carry it as ``custom:<name>``, and a bare agent may carry the name
    in ``provider`` alone.  All of those name one route, so the trailer announces the NAME on
    every path, in one canonical spelling; an unnamed route keeps its collapsed id
    (``custom``, ``auto``, ...), and an agent with no provider announces nothing.
    """
    provider = str(getattr(agent, "provider", "") or "").strip()
    if not provider:
        return ""
    requested = str(getattr(agent, "requested_provider", "") or "").strip()
    return (canonical_provider_value(_provider_name(provider) or _provider_name(requested))
            or canonical_provider_value(provider))


def provider_identity_values(agent: Any) -> set:
    """Every ``Provider:`` value a stored prompt may legitimately carry for THIS route.

    One route has more than one spelling by construction — the configured name, the collapsed
    wire name ``custom``, and the ``custom:<name>`` id a provider picker assigns — so all of
    them are accepted for an unchanged route.  ``custom`` is accepted only for an agent
    actually wired to the custom provider: treating it as an alias of every route would let an
    OpenRouter / Anthropic / MoA agent accept a stored ``Provider: custom``, and a real
    custom -> cloud switch sharing one model string would then never rebuild.

    Deliberately not derived from the provider registry or config: those are absent under a
    sandboxed home and would silently change prompt bytes (and rebuilds) with the environment.
    """
    provider = str(getattr(agent, "provider", "") or "").strip()
    if not provider:
        return set()
    requested = str(getattr(agent, "requested_provider", "") or "").strip()
    values = {runtime_provider_label(agent), canonical_provider_value(provider),
              canonical_provider_value(requested)} - {""}
    if _is_custom_wire(provider):
        values.add("custom")
    return values


def provider_line_matches(stored_value: str, agent: Any) -> bool:
    """True when a stored ``Provider:`` value names this agent's route."""
    stored = canonical_provider_value(stored_value)
    return not stored or stored in provider_identity_values(agent)


def canonical_provider_value(value: str) -> str:
    """One spelling for a provider id or display name: ``custom:Strata Mint`` -> ``strata-mint``.

    ``providers.<key>`` entries are addressed by their key, their display ``name:`` and their
    ``custom:<slug>`` id; without this a session on an entry whose name has a space would
    announce two identities for one route and rebuild on every hand-off between them.
    """
    name = _provider_name(value) or (value or "").strip()
    return name.strip().lower().replace(" ", "-")


def _provider_name(value: str) -> str:
    """The named part of a provider id: ``custom:foo`` -> ``foo``; a collapsed id -> ``""``."""
    value = (value or "").strip()
    lowered = value.lower()
    if lowered.startswith("custom:"):
        return value.split(":", 1)[1].strip()
    return "" if lowered in _COLLAPSED_PROVIDER_LABELS else value


def _is_custom_wire(value: str) -> bool:
    """True for the generic custom wire: ``custom`` or ``custom:<name>``."""
    lowered = (value or "").strip().lower()
    return lowered == "custom" or lowered.startswith("custom:")


def _last_announced_surface(conversation_history: Any) -> str:
    """The surface named by the NEWEST switch note in the transcript ("" when none).

    Once a switch has been announced, that note — not the stored prompt's ``Platform:`` trailer —
    is the last thing the model was told it runs on.  Reading it back is also what keeps a fresh
    AIAgent per turn (the gateway shape) from stacking one copy of the note per turn."""
    for msg in reversed((conversation_history or [])[-_NOTE_SCAN_TAIL:]):
        # The note only ever lands on a user row: in its api_content sidecar, or as a text part
        # when the content is a multimodal list (which cannot take the string sidecar).
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        sidecar = msg.get("api_content")
        text = (sidecar if isinstance(sidecar, str) else "") + "\n" + flatten_message_text(msg.get("content"))
        if _SURFACE_SWITCH_NOTE_PREFIX in text:
            tail = text.rsplit(_SURFACE_SWITCH_NOTE_PREFIX, 1)[1]
            return tail.split(_SURFACE_NAME_END, 1)[0].strip()
    return ""


def note_inert_pinned_tools(agent: Any, built_for_this_surface: list[str]) -> None:
    """Name, at the end of the staged note, the pinned tools THIS surface did not build.

    The freeze keeps a previous surface's tools on the wire deliberately — removing them is the one
    thing that would still re-prefill the request behind the preserved prompt — so the model has
    to be TOLD they are inert here, or it plans around a ``focus_pane`` a terminal turn can only
    answer with ``tool_error("desktop only")``."""
    from tools.mcp_tool_agent import agent_tool_names
    surface_names = set(built_for_this_surface)
    inert = [name for name in agent_tool_names(agent) if name not in surface_names]
    note = getattr(agent, "_surface_switch_note", "") or ""
    if not inert or not note:
        return
    agent._surface_switch_note = note + (
        "\n[System: These tools stay listed for this conversation (dropping them would discard "
        "the cached request prefix) but were not loaded for this interface — expect a call to "
        f"one of them to fail: {', '.join(inert)}.]"
    )


def stage_surface_switch_note(agent: Any, prompt: str, conversation_history: Any) -> bool:
    """Stage a one-shot correction when the request would otherwise misdescribe the surface.

    Compares the runtime surface against what the model was last told — the newest switch note
    if one exists, else ``prompt``'s own ``Platform:`` trailer.  Consulting the note matters in
    both directions: switching BACK to the prompt's surface (desktop -> tui -> desktop) leaves the
    trailer agreeing with the runtime while a stale note still says otherwise, and a rebuild for
    an unrelated reason (a model switch) refreshes the prompt but not that note.  The full surface
    guidance is attached only when the prompt itself is out of date; otherwise the note just
    retires the stale one.  Returns whether it staged.

    MoA and codex_app_server turns never stamp the ``api_content`` sidecar, so the note could not
    be read back and would be re-sent every turn; those modes keep the stored prompt and skip the
    note entirely."""
    if getattr(agent, "provider", None) == "moa" or getattr(agent, "api_mode", None) == "codex_app_server":
        return False
    current = str(getattr(agent, "platform", "") or "").strip()
    if not current:
        return False
    described = identity_line_value(prompt, "Platform")
    told = _last_announced_surface(conversation_history) or described
    if not told or told == current:
        return False
    from agent.system_prompt import platform_hint
    hint = platform_hint(agent) if described != current else ""
    where = "the guidance below" if hint else "the interface section in the system prompt above"
    note = (
        f"{_SURFACE_SWITCH_NOTE_PREFIX}{current}{_SURFACE_NAME_END} in this conversation is "
        f"superseded — follow {where} for formatting, file delivery and any interface-specific "
        "capability.]"
    )
    agent._surface_switch_note = f"{note}\n{hint}" if hint else note
    logger.info(
        "Session %s switched surface %s -> %s; keeping the stored system prompt and delivering "
        "the new surface guidance as a turn note (prefix cache preserved).",
        agent.session_id, told, current,
    )
    return True
