"""Hybrid keyword/BM25 retrieval for the memory store: FTS5 candidates reranked with
Jaccard similarity and HRR vector similarity, trust-weighted (ported from KIK memory_agent.py)."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

from . import holographic as hrr

_FACT_COLUMNS = "fact_id, content, category, tags, trust_score, retrieval_count, helpful_count, created_at, updated_at"
_ROLE_ENTITY, _ROLE_CONTENT = hrr.ROLE_ENTITY, hrr.ROLE_CONTENT
_PUNCT = ".,;:!?\"'()[]{}#@<>"
_FTS_OPERATORS = str.maketrans("", "", '"()*^:-+')
# Stopwords dropped before FTS5 OR-expansion: short English function words that
# carry no retrieval signal and force false-negative AND matches.
_FTS_STOPWORDS = frozenset("""
    a about above after again all am an and any are as at be because been before being between both but by can could
    did do does doing don down during each few for from further had has have having he her here hers herself him himself
    his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other our
    ours ourselves out over own same she should so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves""".split())


def _shift(sim: float) -> float:
    """Cosine similarity [-1, 1] -> [0, 1]."""
    return (sim + 1.0) / 2.0


class FactRetriever:
    """Multi-strategy fact retrieval with trust-weighted scoring."""

    def __init__(self, store: MemoryStore, temporal_decay_half_life: int = 0,  # days, 0 = disabled
                 fts_weight: float = 0.45, jaccard_weight: float = 0.40, hrr_weight: float = 0.15, hrr_dim: int = 1024):
        self.store, self.half_life, self.hrr_dim = store, temporal_decay_half_life, hrr_dim
        if hrr_weight > 0 and not hrr._HAS_NUMPY:  # redistribute weights without numpy
            fts_weight, jaccard_weight, hrr_weight = 0.6, 0.4, 0.0
        self.fts_weight, self.jaccard_weight, self.hrr_weight = fts_weight, jaccard_weight, hrr_weight

    def _atom(self, word: str):
        return hrr.encode_atom(word, self.hrr_dim)

    def _phases(self, blob: bytes):
        return hrr.bytes_to_phases(blob, dim=self.hrr_dim)

    def search(self, query: str, category: str | None = None, min_trust: float = 0.3, limit: int = 10) -> list[dict]:
        """FTS5 candidates (limit*3) → coverage + HRR rerank → trust weighting → optional temporal decay
        0.5^(age_days / half_life). Returns fact dicts with 'score', sorted desc.

        Trust is applied as a neutral-at-0.5 multiplier (0.6 + 0.8*t) rather than a raw factor, so the
        default 0.5 trust leaves the score untouched instead of halving every result. Facts whose body
        or tags mark them closed/superseded are demoted hard so a live position outranks its tombstone.
        """
        query_tokens = self._tokenize(query)
        if "hub" in query_tokens:  # explicit index request: resolve the domain hub directly
            hubs = self._hub_facts(query_tokens, category)
            if hubs:
                for h in hubs:
                    h["score"] = 1.0
                return hubs[:limit]
        candidates = self._fts_candidates(query, category, min_trust, limit * 3)
        # Query vector is loop-invariant; encode lazily on the first candidate that carries an HRR vector
        # so stores whose hrr_vector was never backfilled don't pay for it.
        query_vec = None
        for fact in candidates:
            cover = self._coverage_similarity(query_tokens, self._tokenize(fact["content"]) | self._tokenize(fact.get("tags", "")))
            hrr_sim = 0.5  # neutral
            if self.hrr_weight > 0 and fact.get("hrr_vector"):
                fact_vec = self._phases(fact["hrr_vector"])
                if query_vec is None:
                    query_vec = hrr.encode_text(query, self.hrr_dim)
                hrr_sim = _shift(hrr.similarity(query_vec, fact_vec))
            # fts_rank is normalised inside the candidate set, so a weak 1-of-5-token match scores as
            # high as a strong one. Damping it by query coverage keeps junk hits near 0.2 instead of 0.5.
            relevance = (self.fts_weight * fact.get("fts_rank", 0.0) * (0.7 + 0.3 * cover)
                         + self.jaccard_weight * cover + self.hrr_weight * hrr_sim)
            # "X hub" means X's index fact. Boost only the hub whose own domain is named in the query,
            # otherwise the life hub wins every other domain's hub query (it lists them all).
            tags = fact.get("tags") or ""
            # The path tag encodes the topic: a query token that IS a path segment ("gestor",
            # "funding", "43509") is a far stronger signal than a mention in prose.
            path = next((t for t in tags.split(",") if "." in t), "")
            seg_hits = sum(1 for s in path.split(".") if s and s.lower() in query_tokens)
            if seg_hits:
                relevance += min(0.20, 0.10 * seg_hits)
            if "hub" in query_tokens and ".hub" in tags:
                domain = tags.split(",")[0].strip().lower()
                if domain in query_tokens:
                    relevance += 0.30
            fact["score"] = min(1.0, relevance * (0.6 + 0.8 * fact["trust_score"]))
            if self._is_closed(fact):
                fact["score"] *= 0.35
            if self.half_life > 0:
                fact["score"] *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))
        results = sorted(candidates, key=lambda x: x["score"], reverse=True)[:limit]
        for fact in results:
            fact.pop("hrr_vector", None)  # callers expect JSON-serializable dicts
        return results

    def _vector_query(self, fallback: str, category: str | None, limit: int, sim_fn: Callable) -> list[dict]:
        """Rank every fact vector (optionally per category) by sim_fn; FTS5 fallback when no vectors exist."""
        rows = self._vector_rows(category)
        return self._rank_by_vector(rows, sim_fn, limit) if rows else self.search(fallback, category=category, limit=limit)

    def probe(self, entity: str, category: str | None = None, limit: int = 10) -> list[dict]:
        """Facts that MENTION an entity, straight from the resolved entity table.

        Entity-table lookup is the reliable structural signal in this store: the HRR residual test
        returned noise (similarity ~0) for every entity because a 1 KB fact bundles far more tokens
        than the sqrt(dim)=32 capacity, so the encoding carries no recoverable role structure. When
        the entity has no links, fall back to text search.
        """
        rows = self._facts_for_entities([entity], category)
        if not rows:
            return self.search(entity, category=category, limit=limit)
        scored = self._score_entity_rows(rows, [entity], category)
        return scored[:limit]

    def related(self, entity: str, category: str | None = None, limit: int = 10) -> list[dict]:
        """Facts structurally connected to an entity: share at least one other resolved entity with
        a fact that mentions it. Ranked by number of shared entities, then trust."""
        conn = self.store._conn
        eid = self._resolve_entity_id(entity)
        if eid is None:
            return self.search(entity, category=category, limit=limit)
        # Co-occurrence weighted by entity rarity (IDF): "Relja" links 100+ facts and carries no
        # signal, while "Pupil Labs Neon" or "Tobii" links a handful and is exactly the connection wanted.
        own = {r["fact_id"] for r in conn.execute("SELECT fact_id FROM fact_entities WHERE entity_id = ?", (eid,))}
        if not own:
            return self.search(entity, category=category, limit=limit)
        # Stay inside the entity's own domain: otherwise a MUIA entity links through a rare shared
        # entity to Mira facts, and "related" returns a different project entirely.
        import collections
        dom_counts = collections.Counter()
        for r in conn.execute(f"SELECT tags FROM facts WHERE fact_id IN ({','.join('?' * len(own))})", list(own)):
            dom_counts[(r["tags"] or "").split(",")[0].strip()] += 1
        domain = dom_counts.most_common(1)[0][0] if dom_counts else ""
        marks = ",".join("?" * len(own))
        links = conn.execute(f"SELECT fact_id, entity_id FROM fact_entities WHERE fact_id IN ({marks})",
                             list(own)).fetchall()
        # Very common entities (Relja in 100+ facts, Mira in 40) connect everything to everything:
        # they are pruned so "related" returns a real structural neighbour, not a hub.
        ent_ids = sorted({r["entity_id"] for r in links} - {eid})
        if not ent_ids:
            return []
        df = {r["entity_id"]: r["c"] for r in conn.execute(
            f"SELECT entity_id, COUNT(*) c FROM fact_entities WHERE entity_id IN ({','.join('?' * len(ent_ids))}) "
            "GROUP BY entity_id", ent_ids)}
        ent_ids = [e for e in ent_ids if df.get(e, 0) <= 20]
        if not ent_ids:
            return []
        weights = {e: 1.0 / math.log(2 + c) for e, c in df.items()}
        cand = conn.execute(
            f"SELECT fe.fact_id, fe.entity_id FROM fact_entities fe JOIN facts f ON f.fact_id = fe.fact_id "
            f"WHERE fe.entity_id IN ({','.join('?' * len(ent_ids))}) AND f.tags LIKE ? "
            f"AND fe.fact_id NOT IN ({marks})", ent_ids + [f"{domain},%"] + list(own)).fetchall()
        scores: dict[int, float] = {}
        for r in cand:
            if r["entity_id"] in weights:
                scores[r["fact_id"]] = scores.get(r["fact_id"], 0.0) + weights[r["entity_id"]]
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit * 2]
        facts = self._fetch_facts([i for i, _ in top], category)
        for f in facts:
            f["score"] = round(min(1.0, scores.get(f["fact_id"], 0.0) / max(1.0, max(scores.values())))
                               * (0.6 + 0.8 * f["trust_score"]), 4)
        return sorted(facts, key=lambda x: x["score"], reverse=True)[:limit]

    def reason(self, entities: list[str], category: str | None = None, limit: int = 10) -> list[dict]:
        """Facts where ALL listed entities are mentioned (vector-space JOIN, implemented on the
        entity table). Falls back to ranking by how many of the entities matched."""
        if not entities:
            return []
        rows = self._facts_for_entities(entities, category)
        if not rows:
            return self.search(" ".join(entities), category=category, limit=limit)
        return self._score_entity_rows(rows, entities, category)[:limit]

    def contradict(self, category: str | None = None, threshold: float = 0.3, limit: int = 10) -> list[dict]:
        """Pairs of facts sharing entities (same subject) with low content-vector similarity (different claims). Empty without numpy."""
        if not hrr._HAS_NUMPY:
            return []
        rows = self._vector_rows(category, columns="fact_id, content, category, tags, trust_score, created_at, updated_at, hrr_vector")
        if len(rows) < 2:
            return []
        if len(rows) > 500:  # O(n²) guard: only compare the most recently updated facts
            rows = sorted(rows, key=lambda r: r["updated_at"] or r["created_at"], reverse=True)[:500]
        facts = []  # (public dict, lower-cased entity names, phase vector)
        for row in rows:
            fact = dict(row)
            entity_rows = self.store._conn.execute(
                "SELECT e.name FROM entities e JOIN fact_entities fe ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?",
                (fact["fact_id"],),
            ).fetchall()
            facts.append((fact, {r["name"].lower() for r in entity_rows}, self._phases(fact.pop("hrr_vector"))))
        contradictions = []
        for i, (f1, ents1, vec1) in enumerate(facts):
            for f2, ents2, vec2 in facts[i + 1:]:
                if not ents1 or not ents2:
                    continue
                entity_overlap = len(ents1 & ents2) / len(ents1 | ents2)
                if entity_overlap < 0.3:
                    continue  # not enough shared subject to be contradictory
                content_sim = hrr.similarity(vec1, vec2)
                contradiction_score = entity_overlap * (1.0 - _shift(content_sim))  # high overlap + low similarity
                if contradiction_score >= threshold:
                    contradictions.append({
                        "fact_a": f1, "fact_b": f2,
                        "entity_overlap": round(entity_overlap, 3),
                        "content_similarity": round(content_sim, 3),
                        "contradiction_score": round(contradiction_score, 3),
                        "shared_entities": sorted(ents1 & ents2),
                    })
        return sorted(contradictions, key=lambda x: x["contradiction_score"], reverse=True)[:limit]

    def _vector_rows(self, category: str | None, columns: str = _FACT_COLUMNS + ", hrr_vector") -> list:
        """All facts that carry an HRR vector, optionally filtered by category."""
        where = "WHERE hrr_vector IS NOT NULL" + (" AND category = ?" if category else "")
        return self.store._conn.execute(f"SELECT {columns} FROM facts {where}", [category] if category else []).fetchall()

    def _rank_by_vector(self, rows: list, sim_fn: Callable[[dict, object], float], limit: int) -> list[dict]:
        """Score each row as (sim + 1) / 2 * trust-neutral weight (0.6 + 0.8*trust), sorted desc."""
        scored = [dict(row) for row in rows]
        for fact in scored:
            fact["score"] = _shift(sim_fn(fact, self._phases(fact.pop("hrr_vector")))) * (0.6 + 0.8 * fact["trust_score"])
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:limit]

    def _fts_candidates(self, query: str, category: str | None, min_trust: float, limit: int) -> list[dict]:
        """Raw FTS5 MATCH candidates with rank normalized to [0, 1] as 'fts_rank'."""
        category_clause = "AND f.category = ? " if category else ""
        params = [self._sanitize_fts_query(query)] + ([category] if category else []) + [min_trust, limit]
        sql = ("SELECT f.*, facts_fts.rank as fts_rank_raw FROM facts_fts JOIN facts f ON f.fact_id = facts_fts.rowid "
               f"WHERE facts_fts MATCH ? {category_clause}AND f.trust_score >= ? ORDER BY facts_fts.rank LIMIT ?")
        try:
            results = [dict(row) for row in self.store._conn.execute(sql, params).fetchall()]
        except Exception:
            return []  # FTS5 MATCH can fail on malformed queries
        # FTS5 rank is negative (lower = better); normalize |rank| / max to [0, 1] (1e-6 floor avoids div by zero)
        max_rank = max([abs(f["fts_rank_raw"]) for f in results] + [1e-6])
        for fact in results:
            fact["fts_rank"] = abs(fact.pop("fts_rank_raw")) / max_rank
        return results

    # --- entity-table helpers (probe / related / reason) -------------------------------------------

    def _hub_facts(self, tokens: set[str], category: str | None) -> list[dict]:
        """Facts tagged <domain>.hub where <domain> is one of the query tokens."""
        out: list[dict] = []
        for tok in tokens:
            rows = self.store._conn.execute(
                "SELECT fact_id FROM facts WHERE tags LIKE ? AND tags LIKE '%.hub%'", (f"{tok},%",)).fetchall()
            out += self._fetch_facts([r["fact_id"] for r in rows], category)
        return out

    def _resolve_entity_id(self, name: str) -> int | None:
        row = self.store._conn.execute("SELECT entity_id FROM entities WHERE name = ? COLLATE NOCASE", (name.strip(),)).fetchone()
        return int(row["entity_id"]) if row else None

    def _facts_for_entities(self, names: list[str], category: str | None) -> list[dict]:
        """Facts mentioning ANY of the given entities, with a matched_entities count."""
        ids = [self._resolve_entity_id(n) for n in names]
        ids = [i for i in ids if i is not None]
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        sql = (f"SELECT fe.fact_id, COUNT(DISTINCT fe.entity_id) AS matched FROM fact_entities fe "
               f"WHERE fe.entity_id IN ({marks}) GROUP BY fe.fact_id")
        params: list = list(ids)
        if category:
            sql = sql.replace("GROUP BY", "AND fe.fact_id IN (SELECT fact_id FROM facts WHERE category = ?) GROUP BY")
            params.append(category)
        rows = [(r["fact_id"], r["matched"]) for r in self.store._conn.execute(sql, params).fetchall()]
        facts = self._fetch_facts([i for i, _ in rows], category)
        counts = dict(rows)
        for f in facts:
            f["matched_entities"] = counts.get(f["fact_id"], 0)
        return facts

    def _score_entity_rows(self, rows: list[dict], names: list[str], category: str | None) -> list[dict]:
        """Rank entity-linked facts: mention coverage of the queried entity set, trust, and whether the
        entity name also appears in the text/tags (a true mention, not a regex artefact)."""
        need = max(1, len({n.strip().lower() for n in names}))
        for f in rows:
            text = f"{f['content']} {f.get('tags', '')}".lower()
            literal = sum(1 for n in names if n.strip().lower() in text) / need
            coverage = f.get("matched_entities", 0) / need
            # Density separates a fact ABOUT the entity from a fact that merely mentions it: an
            # entity mentioned many times in a short fact is the subject; one mention in a long
            # roundup is incidental. Without it every linked fact ties at 1.0.
            hits = sum(text.count(n.strip().lower()) for n in names)
            density = min(1.0, hits / max(2.0, len(text.split()) / 40.0))
            focus = 1.0 / (1.0 + max(0, len(text) - 800) / 2000.0)
            base = min(1.0, 0.40 * coverage + 0.30 * literal + 0.20 * density + 0.10 * focus) * (0.6 + 0.8 * f["trust_score"])
            if self._is_closed(f):
                base *= 0.35
            f["score"] = round(min(1.0, base), 4)
        return sorted(rows, key=lambda x: (x["score"], x.get("updated_at") or ""), reverse=True)

    def _fetch_facts(self, ids: list[int], category: str | None) -> list[dict]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        sql = f"SELECT {_FACT_COLUMNS} FROM facts WHERE fact_id IN ({marks})"
        params: list = list(ids)
        if category:
            sql += " AND category = ?"
            params.append(category)
        return [dict(r) for r in self.store._conn.execute(sql, params).fetchall()]

    @staticmethod
    def _is_closed(fact: dict) -> bool:
        """A tombstone fact: body opens with CLOSED/SUPERSEDED or a tag says so."""
        tags = (fact.get("tags") or "").lower()
        if "closed" in tags.split(",") or "superseded" in tags.split(","):
            return True
        head = (fact.get("content") or "").lstrip().lower()
        return head.startswith("closed") or head.startswith("superseded") or "superseded by fact" in head[:200]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Lowercase tokens with surrounding punctuation stripped, with light plural folding so
        'cat'/'cats', 'deadline'/'deadlines', 'policy'/'policies' meet. No full stemmer."""
        if not text:
            return set()
        out = set()
        for raw in text.lower().split():
            tok = raw.strip(_PUNCT)
            if not tok:
                continue
            folded = tok
            if len(tok) > 3 and tok.endswith("ies"):
                folded = tok[:-3] + "y"
            elif len(tok) > 4 and tok.endswith("es") and not tok.endswith("sses"):
                folded = tok[:-2]
            elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
                folded = tok[:-1]
            out.add(tok)          # keep the surface form: 'hermes' must not collapse to 'herme'
            out.add(folded)
        return out


    def _sanitize_fts_query(self, query: str) -> str:
        """Natural-language query -> FTS5-safe OR expression of quoted tokens. FTS5 AND-joins a multi-word
        MATCH by default, which tanks recall on prose: drop stopwords and <2-char tokens, strip FTS5 operator
        chars, phrase-quote each survivor. If nothing survives, return the raw query (zero results, not a SQL error)."""
        if not query:
            return ""
        # Prefix-match plain words so plurals meet ('deadline' -> 'deadlines'). Dotted path tags must stay
        # quoted phrases: a bare 'mira.tech.ml*' is not a valid FTS5 bareword and the whole MATCH raises,
        # silently returning zero results for every path-tag query.
        tokens = []
        for raw in query.lower().split():
            if "." in raw or "-" in raw:  # path/entity tags: keep '.' and '-' inside a quoted phrase
                c = raw.strip(_PUNCT).replace('"', "")
            else:
                # Apostrophes and other inner punctuation are NOT FTS5 bareword characters: "gestor's*"
                # is a syntax error, and the swallowed exception silently zeroed the whole query.
                c = re.sub(r"[^a-z0-9]", "", raw.strip(_PUNCT).translate(_FTS_OPERATORS))
            if len(c) < 2 or c in _FTS_STOPWORDS:
                continue
            if "." in c or "-" in c or len(c) < 3:
                tokens.append(f'"{c}"')
            elif self._fts_has_exact(c):
                tokens.append(f'"{c}"')
            else:  # 'cat' -> 'cats*', 'medicat' -> only if the exact word is absent
                tokens.append(f'{c}*')
        return " OR ".join(tokens) if tokens else query

    def _fts_has_exact(self, token: str) -> bool:
        """True when the store contains this exact word (so no prefix expansion is needed)."""
        try:
            row = self.store._conn.execute("SELECT 1 FROM facts_fts WHERE facts_fts MATCH ? LIMIT 1",
                                           (f'"{token}"',)).fetchone()
            return row is not None
        except Exception:
            return True

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|."""
        return len(set_a & set_b) / len(set_a | set_b) if set_a and set_b else 0.0

    @staticmethod
    def _coverage_similarity(query_tokens: set[str], fact_tokens: set[str]) -> float:
        """How much of the QUERY the fact covers, with a mild length damping.

        Plain Jaccard divides by |A∪B|, so a 4 KB hub or any long fact is punished for being long:
        the hub ranked 4th for 'Mira hub'. Here the denominator is the query size plus a small
        fraction of the excess fact tokens — long facts are mildly penalised, not linearly.
        """
        if not query_tokens or not fact_tokens:
            return 0.0
        overlap = len(query_tokens & fact_tokens)
        if not overlap:
            return 0.0
        excess = max(0, len(fact_tokens) - len(query_tokens))
        return min(1.0, overlap / (len(query_tokens) + 0.04 * excess))


    def _temporal_decay(self, timestamp_str: str | None) -> float:
        """0.5^(age_days / half_life); 1.0 if disabled, missing, unparseable, or in the future."""
        if not self.half_life or not timestamp_str:
            return 1.0
        try:
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")) if isinstance(timestamp_str, str) else timestamp_str
            age_days = (datetime.now(timezone.utc) - (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc))).total_seconds() / 86400
            return 1.0 if age_days < 0 else math.pow(0.5, age_days / self.half_life)
        except (ValueError, TypeError):
            return 1.0
