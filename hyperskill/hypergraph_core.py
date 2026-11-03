#!/usr/bin/env python3
"""Hypergraph memory core for HyperSkill."""

import os
import json
import math
import hashlib
import logging
from typing import List, Dict, Tuple, Optional, Any, Set

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


SUBTASK = "subtask"
SKILL = "skill"
MISTAKE = "mistake"
EPISODE = "episode"
def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                         cache_dir: str = "./storage/models"):
    """Load SentenceTransformer model with local caching. Respects EMBEDDING_DEVICE."""
    from sentence_transformers import SentenceTransformer

    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, model_name.replace("/", "_"))
    device = os.environ.get("EMBEDDING_DEVICE", None)

    try:
        if os.path.exists(local_path) and os.listdir(local_path):
            return SentenceTransformer(local_path, device=device)
    except Exception as e:
        logger.warning("Local model load failed: %s", e)

    model = SentenceTransformer(model_name, device=device)
    try:
        model.save(local_path)
    except Exception:
        pass
    return model

class EmbeddingIndex:
    """In-memory embedding index backed by JSONL for persistence."""

    def __init__(self, path: str):
        self.path = path
        self._ids: List[str] = []
        self._vecs: List[List[float]] = []
        self._id_set: Set[str] = set()
        self._matrix: Optional[np.ndarray] = None
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    eid = obj["id"]
                    if eid not in self._id_set:
                        self._ids.append(eid)
                        self._vecs.append(obj["embedding"])
                        self._id_set.add(eid)
                except (json.JSONDecodeError, KeyError):
                    continue
        self._invalidate()

    def rewrite(self):
        """Rewrite the full JSONL file from in-memory state."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            for eid, vec in zip(self._ids, self._vecs):
                f.write(json.dumps({"id": eid, "embedding": vec}) + "\n")

    def _invalidate(self):
        self._matrix = None

    def _ensure_matrix(self):
        if self._matrix is not None:
            return
        if not self._vecs:
            self._matrix = None
            return
        mat = np.array(self._vecs, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._matrix = mat / norms

    def add(self, eid: str, text: str, vec: np.ndarray):
        if eid in self._id_set:
            return
        v = vec.flatten().tolist()
        self._ids.append(eid)
        self._vecs.append(v)
        self._id_set.add(eid)
        self._invalidate()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({"id": eid, "text": text, "embedding": v}) + "\n")

    def remove(self, eid: str):
        if eid not in self._id_set:
            return
        idx = self._ids.index(eid)
        self._ids.pop(idx)
        self._vecs.pop(idx)
        self._id_set.discard(eid)
        self._invalidate()

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = v.flatten().astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def topk(self, query_vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Return top-k (id, cosine_score) pairs."""
        if not self._vecs:
            return []
        self._ensure_matrix()
        qv = self._normalize(query_vec)
        scores = self._matrix @ qv
        k = min(k, len(self._ids))
        if k <= 0:
            return []
        if k >= len(scores):
            top_idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, k)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self._ids[i], float(scores[i])) for i in top_idx]

    def get_score(self, query_vec: np.ndarray, node_id: str) -> float:
        """Cosine similarity between query and a specific node."""
        if node_id not in self._id_set:
            return 0.0
        idx = self._ids.index(node_id)
        vec = np.array(self._vecs[idx], dtype=np.float32)
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
        qv = self._normalize(query_vec)
        return float(np.dot(qv, vec))

    def find_similar_by_vec(self, query_vec: np.ndarray,
                            node_type_filter: Optional[str],
                            graph: nx.Graph,
                            threshold: float) -> Optional[str]:
        """Find first node above threshold with matching type."""
        if not self._vecs:
            return None
        self._ensure_matrix()
        qv = self._normalize(query_vec)
        scores = self._matrix @ qv
        order = np.argsort(-scores)
        for idx in order:
            if scores[idx] < threshold:
                break
            nid = self._ids[idx]
            if node_type_filter:
                nd = graph.nodes.get(nid, {})
                if nd.get("type") != node_type_filter:
                    continue
            return nid
        return None

    def get_vec(self, eid: str) -> Optional[np.ndarray]:
        if eid not in self._id_set:
            return None
        idx = self._ids.index(eid)
        return np.array(self._vecs[idx], dtype=np.float32)

    def get_vecs(self, eids: List[str]) -> np.ndarray:
        dim = len(self._vecs[0]) if self._vecs else 384
        vecs = []
        for eid in eids:
            if eid in self._id_set:
                idx = self._ids.index(eid)
                vecs.append(self._vecs[idx])
            else:
                vecs.append([0.0] * dim)
        return np.array(vecs, dtype=np.float32)

    def __contains__(self, eid: str) -> bool:
        return eid in self._id_set

    def __len__(self) -> int:
        return len(self._ids)

class HyperGraph:
    """Hypergraph memory system.

    Bipartite representation:
        - Regular nodes: subtask / skill / mistake  (role="node")
        - Hyperedge nodes: episode  (role="hyperedge")
        - Edges: hyperedge_node <-> member_nodes  (role="membership")
    """

    def __init__(self, base_dir: str = "./memories",
                 embedding_model=None,
                 config: Optional[dict] = None):
        os.makedirs(base_dir, exist_ok=True)
        self._base_dir = base_dir
        self._embedding_model = embedding_model
        self._config = config or {}
        self._graph_path = os.path.join(base_dir, "graph.graphml")

        self._beta = float(self._config.get("beta", 0.7))
        self._skill_dedup_threshold = float(
            self._config.get("skill_semantic_dedup_threshold", 0.90)
        )

        self.G = nx.Graph()
        if os.path.exists(self._graph_path):
            self.G = nx.read_graphml(self._graph_path)
            logger.info("loaded graph: %d nodes, %d edges",
                        self.G.number_of_nodes(), self.G.number_of_edges())

        self.node_emb = EmbeddingIndex(os.path.join(base_dir, "node_embeddings.jsonl"))
        self.hyperedge_emb = EmbeddingIndex(os.path.join(base_dir, "hyperedge_embeddings.jsonl"))

        self._global_t_min = float("inf")
        self._global_t_max = 0.0
        self._init_global_step_bounds()

        self._dirty = False


    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if self._embedding_model is None:
            raise RuntimeError("No embedding model configured")
        return self._embedding_model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False,
        )

    def set_embedding_model(self, model):
        self._embedding_model = model


    @staticmethod
    def _ensure_graphml_writable():
        """networkx < 3.4 builds its GraphML type table from ``np.float_`` /
        ``np.int_`` aliases that NumPy 2 removed, so ``write_graphml`` raises
        before it ever looks at the data. Restore the aliases just for that
        lookup; a correctly pinned install (networkx >= 3.4) never hits this.
        """
        for alias, target in (("float_", "float64"), ("int_", "int64"),
                              ("complex_", "complex128")):
            if not hasattr(np, alias) and hasattr(np, target):
                setattr(np, alias, getattr(np, target))

    def save(self):
        """Persist graph to disk (GraphML)."""
        self._ensure_graphml_writable()
        for nid in self.G.nodes:
            for k, v in list(self.G.nodes[nid].items()):
                if isinstance(v, (dict, list)):
                    self.G.nodes[nid][k] = str(v)
                elif v is None:
                    self.G.nodes[nid][k] = ""
        nx.write_graphml(self.G, self._graph_path)
        self._dirty = False

    def flush(self):
        if self._dirty:
            self.save()


    def _init_global_step_bounds(self):
        """Scan existing nodes to initialize T_min, T_max.

        T_min stays at +inf when no node has been observed yet, so that the
        first real observation sets it (initialising it to 0.0 here would pin
        T_min to 0 forever, since min(0, T) == 0 for any T >= 0).
        _compute_utility_score treats an unset T_min as "no range yet".
        """
        for nid in self.G.nodes:
            nd = self.G.nodes[nid]
            if nd.get("role") != "node":
                continue
            rc = int(float(nd.get("retrieval_count", 0)))
            avg_s = float(nd.get("avg_steps", 0))
            if rc > 0 and avg_s > 0:
                self._global_t_min = min(self._global_t_min, avg_s)
                self._global_t_max = max(self._global_t_max, avg_s)

    def _compute_utility_score(self, success_count: int, retrieval_count: int,
                               avg_steps: float) -> float:
        """gamma = beta*(sigma/nu) + (1-beta)*(1 - (T_avg-T_min)/(T_max-T_min))

        For untested nodes (nu=0): return 0.5 (neutral). Since gamma is only used
        in maintenance pruning, this just means untested nodes are ineligible
        for pruning until they've been tested (rc >= N_min).
        """
        if retrieval_count == 0:
            return 0.5

        success_rate = success_count / retrieval_count
        t_range = (0.0 if math.isinf(self._global_t_min)
                   else self._global_t_max - self._global_t_min)
        if t_range > 0 and avg_steps > 0:
            step_eff = 1.0 - (avg_steps - self._global_t_min) / t_range
            step_eff = max(0.0, min(1.0, step_eff))
        else:
            step_eff = 1.0
        return max(0.0, min(1.0,
                            self._beta * success_rate + (1 - self._beta) * step_eff))

    def update_utility(self, retrieved_node_ids: List[str],
                       success: bool, step_count: int):
        """Update gamma tracking for retrieved nodes + their incident hyperedges.

        Called after each episode. gamma is consumed only by maintain_graph().
        """
        updated_he_ids: Set[str] = set()

        for nid in retrieved_node_ids:
            if not self.G.has_node(nid):
                continue
            nd = self.G.nodes[nid]

            prev_rc = int(float(nd.get("retrieval_count", 0)))
            rc = prev_rc + 1
            sc = int(float(nd.get("success_count", 0))) + (1 if success else 0)
            prev_avg = float(nd.get("avg_steps", 0))
            avg_s = ((prev_avg * prev_rc + step_count) / rc
                     if prev_rc > 0 else float(step_count))

            nd["retrieval_count"] = rc
            nd["success_count"] = sc
            nd["avg_steps"] = avg_s

            if avg_s > 0:
                self._global_t_min = min(self._global_t_min, avg_s)
                self._global_t_max = max(self._global_t_max, avg_s)

            nd["utility"] = self._compute_utility_score(sc, rc, avg_s)

            for he_id in self.get_node_hyperedges(nid):
                updated_he_ids.add(he_id)

        for he_id in updated_he_ids:
            if not self.G.has_node(he_id):
                continue
            he = self.G.nodes[he_id]
            prev_rc = int(float(he.get("retrieval_count", 0)))
            rc = prev_rc + 1
            sc = int(float(he.get("success_count", 0))) + (1 if success else 0)
            prev_avg = float(he.get("avg_steps", 0))
            avg_s = ((prev_avg * prev_rc + step_count) / rc
                     if prev_rc > 0 else float(step_count))
            he["retrieval_count"] = rc
            he["success_count"] = sc
            he["avg_steps"] = avg_s
            he["utility"] = self._compute_utility_score(sc, rc, avg_s)

        self._dirty = True


    def add_subtask(self, name: str, content: str,
                    metadata: Optional[dict] = None) -> str:
        """Add a subtask node as a per-episode anchor.

        Dedup: content-aware exact hash (md5(name + '||' + content)).
        No semantic dedup - subtasks carry instance-specific context.
        """
        name = (name or "").strip()
        content = (content or "").strip()
        if not name:
            return ""

        nid = f"st_{_md5(name + '||' + content)}"
        if self.G.has_node(nid):
            return nid

        attrs = {
            "name": name,
            "content": content,
            "type": SUBTASK,
            "role": "node",
            "retrieval_count": 0,
            "success_count": 0,
            "avg_steps": 0.0,
            "utility": 0.5,
        }
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (dict, list)):
                    attrs[k] = str(v)
                elif v is None:
                    attrs[k] = ""
                else:
                    attrs[k] = v

        self.G.add_node(nid, **attrs)

        embed_text = f"{name}. {content}" if content else name
        vec = self.embed_texts([embed_text])[0]
        self.node_emb.add(nid, embed_text, vec)

        self._dirty = True
        return nid

    def _add_abstract_node(self, name: str, content: str, node_type: str,
                           metadata: Optional[dict] = None) -> str:
        """Add a skill or mistake node with name-MD5 + Layer 2 semantic dedup."""
        name = (name or "").strip()
        content = (content or "").strip()
        if not name:
            return ""

        prefix = "sk" if node_type == SKILL else "mk"
        nid = f"{prefix}_{_md5(name)}"

        if self.G.has_node(nid):
            return nid

        embed_parts = [name]
        if content:
            embed_parts.append(content)
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, str) and v:
                    embed_parts.append(v)
        full_text = ". ".join(embed_parts)

        if self._skill_dedup_threshold > 0 and len(self.node_emb) > 0:
            vec = self.embed_texts([full_text])[0]
            existing = self.node_emb.find_similar_by_vec(
                vec, node_type, self.G, self._skill_dedup_threshold
            )
            if existing is not None:
                return existing
        else:
            vec = self.embed_texts([full_text])[0]

        attrs = {
            "name": name,
            "content": content,
            "type": node_type,
            "role": "node",
            "retrieval_count": 0,
            "success_count": 0,
            "avg_steps": 0.0,
            "utility": 0.5,
        }
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (dict, list)):
                    attrs[k] = str(v)
                elif v is None:
                    attrs[k] = ""
                else:
                    attrs[k] = v

        self.G.add_node(nid, **attrs)
        self.node_emb.add(nid, full_text, vec)

        self._dirty = True
        return nid

    def add_skill(self, name: str, content: str,
                  metadata: Optional[dict] = None) -> str:
        return self._add_abstract_node(name, content, SKILL, metadata)

    def add_mistake(self, name: str, content: str,
                    metadata: Optional[dict] = None) -> str:
        return self._add_abstract_node(name, content, MISTAKE, metadata)

    def get_node(self, node_id: str) -> Optional[dict]:
        if not self.G.has_node(node_id):
            return None
        data = dict(self.G.nodes[node_id])
        data["id"] = node_id
        return data

    def get_nodes_by_type(self, node_type: str) -> List[dict]:
        return [
            {**dict(self.G.nodes[nid]), "id": nid}
            for nid in self.G.nodes
            if self.G.nodes[nid].get("role") == "node"
            and self.G.nodes[nid].get("type") == node_type
        ]


    def add_hyperedge(self, edge_type: str, node_ids: List[str],
                      fragment: str,
                      task_query: str = "",
                      metadata: Optional[dict] = None) -> str:
        """Add a hyperedge.

        embedding is built from `task_query + fragment` (combined).
        Fragment is the 1-2 sentence distilled lesson - used for both embedding
        (discovery) and storage (debugging). Not shown to agent.
        """
        fragment = str(fragment or "").strip()
        task_query = str(task_query or "").strip()

        # One hyperedge per trajectory: the task description is part of the
        # identity, otherwise two tasks that yield the same lesson over the
        # same nodes collapse into one edge and reset its utility counters.
        hid = f"he_{_md5(task_query + '||' + fragment + '||' + str(sorted(node_ids)))}"

        attrs = {
            "type": edge_type,
            "role": "hyperedge",
            "fragment": fragment,
            "task_query": task_query,
            "member_count": len(node_ids),
            "retrieval_count": 0,
            "success_count": 0,
            "avg_steps": 0.0,
            "utility": 0.5,
        }
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (dict, list)):
                    attrs[k] = str(v)
                elif v is None:
                    attrs[k] = ""
                else:
                    attrs[k] = v

        self.G.add_node(hid, **attrs)

        for nid in node_ids:
            if self.G.has_node(nid):
                self.G.add_edge(hid, nid, role="membership")

        parts = []
        if task_query:
            parts.append(task_query)
        if fragment:
            parts.append(fragment)
        embed_text = " ".join(parts) if parts else "empty episode"

        vec = self.embed_texts([embed_text])[0]
        self.hyperedge_emb.add(hid, embed_text, vec)

        self._dirty = True
        return hid

    def get_hyperedge_members(self, he_id: str) -> List[str]:
        if not self.G.has_node(he_id):
            return []
        return [
            nbr for nbr in self.G.neighbors(he_id)
            if self.G.edges[he_id, nbr].get("role") == "membership"
        ]

    def get_node_hyperedges(self, node_id: str) -> List[str]:
        if not self.G.has_node(node_id):
            return []
        return [
            nbr for nbr in self.G.neighbors(node_id)
            if self.G.nodes.get(nbr, {}).get("role") == "hyperedge"
        ]


    def retrieve(
        self,
        task_desc: str,
        plan_subtasks: Optional[List[dict]] = None,
        k: int = 5,
    ) -> Dict[str, Any]:
        """Retrieve skills, mistakes, and hyperedges for a given task."""
        empty = {
            "skills": [], "mistakes": [], "hyperedges": [],
            "retrieved_node_ids": [],
            "debug": {},
        }

        if self.G.number_of_nodes() == 0:
            return empty

        task_vec = self.embed_texts([task_desc])[0]
        plan_vec = None
        if plan_subtasks:
            plan_text = " ".join(
                s.get("description", s.get("name", ""))
                for s in plan_subtasks if isinstance(s, dict)
            ).strip()
            if plan_text:
                plan_vec = self.embed_texts([plan_text])[0]

        k_e = math.ceil(k / 2)
        k_u = k - k_e

        e_task: List[str] = []
        for hid, _sim in self.hyperedge_emb.topk(task_vec, k_e):
            if self.G.has_node(hid) and self.G.nodes[hid].get("role") == "hyperedge":
                e_task.append(hid)

        v_sub: List[str] = []
        if plan_vec is not None and k_u > 0:
            hits = self.node_emb.topk(plan_vec, k_u * 3)
            for nid, _sim in hits:
                if not self.G.has_node(nid):
                    continue
                nd = self.G.nodes[nid]
                if nd.get("role") == "node" and nd.get("type") == SUBTASK:
                    v_sub.append(nid)
                    if len(v_sub) >= k_u:
                        break

        e_sub: Set[str] = set()
        for sid in v_sub:
            for hid in self.get_node_hyperedges(sid):
                e_sub.add(hid)

        all_he_ids: Set[str] = set(e_task) | e_sub

        v_sm: Set[str] = set()
        for hid in all_he_ids:
            for mid in self.get_hyperedge_members(hid):
                if self.G.has_node(mid):
                    nd = self.G.nodes[mid]
                    if nd.get("role") == "node" and nd.get("type") in (SKILL, MISTAKE):
                        v_sm.add(mid)

        v_sub_set: Set[str] = set(v_sub)
        skills, mistakes = self._rank_skills_mistakes(
            v_sm, v_sub_set, all_he_ids, task_vec, k
        )

        hyperedges_out: List[dict] = []
        for hid in all_he_ids:
            if self.G.has_node(hid):
                he_data = dict(self.G.nodes[hid])
                he_data["id"] = hid
                # Similarity to the query, for inspection: hyperedges reached
                # through the subtask path have no trajectory-path score, so
                # recompute it here rather than leaving the field at 0.
                he_data["_sim"] = round(
                    max(self.hyperedge_emb.get_score(task_vec, hid), 0.0), 4)
                hyperedges_out.append(he_data)

        retrieved_node_ids = list(v_sub_set) + [n.get("id", "") for n in skills + mistakes]
        seen: Set[str] = set()
        retrieved_node_ids = [
            x for x in retrieved_node_ids
            if x and not (x in seen or seen.add(x))
        ]

        return {
            "skills": skills,
            "mistakes": mistakes,
            "hyperedges": hyperedges_out,
            "retrieved_node_ids": retrieved_node_ids,
            "debug": {
                "e_task": e_task,
                "e_sub": list(e_sub),
                "all_he_ids": list(all_he_ids),
                "v_sub": v_sub,
                "v_sm": list(v_sm),
            },
        }

    def _rank_skills_mistakes(self, v_sm: Set[str], v_sub: Set[str],
                              all_he_ids: Set[str], task_vec: np.ndarray,
                              k: int) -> Tuple[List[dict], List[dict]]:
        """Rank R_s by kappa(v) = |{e in E* : v in e}|.

        kappa counts the number of retrieved hyperedges containing skill v.
        Ties broken by sim(task_vec, v).
        """
        cooc: Dict[str, int] = {}
        for v in v_sm:
            v_hes = set(self.get_node_hyperedges(v))
            cooc[v] = len(v_hes & all_he_ids)

        scored: List[dict] = []
        for v in v_sm:
            node = self.get_node(v)
            if node is None:
                continue
            sim = max(self.node_emb.get_score(task_vec, v), 0.0)
            c = cooc.get(v, 0)
            node["_sim"] = round(sim, 4)
            node["_co_occurrence"] = c
            node["_rank"] = (c, sim)  # primary: kappa, tiebreaker: sim
            scored.append(node)

        scored.sort(key=lambda n: n.get("_rank", (0, 0)), reverse=True)

        skills = [n for n in scored if n.get("type") == SKILL][:k]
        mistakes = [n for n in scored if n.get("type") == MISTAKE][:k]
        return skills, mistakes

    def maintain_graph(
        self,
        merge_threshold: float = 0.85,
        propagation_depth: int = 2,
        merge_alpha: float = 0.5,
        prune_min_retrievals: int = 3,
        prune_threshold: float = 0.2,
        llm_merge_fn=None,
    ) -> Dict[str, Any]:
        """Periodic maintenance.

        Step A: Quality-driven pruning (Eq.10) - skill/mistake nodes only
        Step B: Structure-informed merging (Eq.8-9) - skill/mistake nodes only
        Step C: Orphan subtask cleanup - remove subtasks with no incident HEs
        """
        pruned = self._prune_low_utility(prune_min_retrievals, prune_threshold)
        merged = self._structural_merge_abstract(
            merge_threshold, propagation_depth, merge_alpha, llm_merge_fn
        )
        orphans = self._cleanup_orphan_subtasks()

        self.flush()
        logger.info("maintenance: pruned=%d, merged=%d, orphans_removed=%d",
                    pruned, merged, orphans)
        return {"pruned": pruned, "merged": merged, "orphans": orphans}

    def _prune_low_utility(self, min_retrievals: int,
                           prune_threshold: float) -> int:
        """Eq.10: remove skill/mistake nodes where rc >= N_min AND gamma < tau_prune."""
        to_remove = []
        for nid in list(self.G.nodes):
            nd = self.G.nodes[nid]
            if nd.get("role") != "node":
                continue
            if nd.get("type") not in (SKILL, MISTAKE):
                continue  # never prune subtasks this way
            rc = int(float(nd.get("retrieval_count", 0)))
            util = float(nd.get("utility", 1.0))
            if rc >= min_retrievals and util < prune_threshold:
                to_remove.append(nid)

        for nid in to_remove:
            nd = self.G.nodes[nid]
            logger.info("  prune: %s '%s' (type=%s, rc=%s, gamma=%s)",
                        nid[:15], nd.get("name", ""), nd.get("type", ""),
                        nd.get("retrieval_count", 0), nd.get("utility", ""))
            content = str(nd.get("content", "") or "").replace("\n", " ")
            logger.info("    pruned_content: %s", content[:500])
            self.G.remove_node(nid)
            self.node_emb.remove(nid)

        if to_remove:
            self.node_emb.rewrite()
            self._dirty = True
        return len(to_remove)

    def _cleanup_orphan_subtasks(self) -> int:
        """Remove subtask nodes with no remaining incident hyperedges."""
        to_remove = []
        for nid in list(self.G.nodes):
            nd = self.G.nodes[nid]
            if nd.get("role") != "node":
                continue
            if nd.get("type") != SUBTASK:
                continue
            if not self.get_node_hyperedges(nid):
                to_remove.append(nid)

        for nid in to_remove:
            self.G.remove_node(nid)
            self.node_emb.remove(nid)

        if to_remove:
            self.node_emb.rewrite()
            self._dirty = True
        return len(to_remove)


    def _build_co_occurrence_matrix(self, nids: List[str]) -> np.ndarray:
        """Quality-weighted co-occurrence W (Eq.8)."""
        n = len(nids)
        nid_to_idx = {nid: i for i, nid in enumerate(nids)}
        W = np.zeros((n, n), dtype=np.float32)

        for he_nid in self.G.nodes:
            if self.G.nodes[he_nid].get("role") != "hyperedge":
                continue
            members = self.get_hyperedge_members(he_nid)
            idxs = [nid_to_idx[m] for m in members if m in nid_to_idx]
            if len(idxs) < 2:
                continue
            he_size = max(len(members), 1)
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    gi = float(self.G.nodes[nids[i]].get("utility", 1.0))
                    gj = float(self.G.nodes[nids[j]].get("utility", 1.0))
                    w = min(gi, gj) / he_size
                    W[i, j] += w
                    W[j, i] += w
        return W

    def _l_step_propagation(self, W: np.ndarray, Phi: np.ndarray,
                            L: int, alpha: float) -> np.ndarray:
        """Eq.9: L-step symmetric propagation with restart."""
        n = W.shape[0]
        if n == 0:
            return Phi

        W_hat = W + np.eye(n, dtype=np.float32)
        d = W_hat.sum(axis=1)
        d_inv_sqrt = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if d[i] > 0:
                d_inv_sqrt[i] = 1.0 / math.sqrt(d[i])
        D_inv_sqrt = np.diag(d_inv_sqrt)
        W_norm = D_inv_sqrt @ W_hat @ D_inv_sqrt

        S = np.zeros((n, n), dtype=np.float32)
        W_power = np.eye(n, dtype=np.float32)
        for l in range(L):
            S += alpha * ((1 - alpha) ** l) * W_power
            W_power = W_power @ W_norm
        S += ((1 - alpha) ** L) * W_power

        Phi_tilde = S @ Phi
        norms = np.linalg.norm(Phi_tilde, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return Phi_tilde / norms

    def _find_merge_candidates(self, threshold: float,
                               propagation_depth: int,
                               merge_alpha: float
                               ) -> List[Tuple[str, str, float]]:
        by_type: Dict[str, List[str]] = {}
        for nid in self.G.nodes:
            nd = self.G.nodes[nid]
            if nd.get("role") != "node":
                continue
            ntype = nd.get("type", "")
            if ntype in (SKILL, MISTAKE):
                by_type.setdefault(ntype, []).append(nid)

        candidates = []
        for ntype, nids in by_type.items():
            if len(nids) < 2:
                continue
            Phi = self.node_emb.get_vecs(nids)
            W = self._build_co_occurrence_matrix(nids)
            Phi_tilde = self._l_step_propagation(
                W, Phi, L=propagation_depth, alpha=merge_alpha
            )
            sims = Phi_tilde @ Phi_tilde.T
            for i in range(len(nids)):
                for j in range(i + 1, len(nids)):
                    sim = float(sims[i, j])
                    if sim >= threshold:
                        candidates.append((nids[i], nids[j], sim))
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    def _merge_nodes(self, node_a: str, node_b: str, merged_attrs: dict):
        """Merge node_b into node_a."""
        if not self.G.has_node(node_a) or not self.G.has_node(node_b):
            return

        nd_a = self.G.nodes[node_a]
        nd_b = self.G.nodes[node_b]

        _nd_b_name_snapshot = nd_b.get("name", "")
        _nd_b_content_snapshot = str(nd_b.get("content", "") or "")

        rc_a = max(int(float(nd_a.get("retrieval_count", 0))), 1)
        rc_b = max(int(float(nd_b.get("retrieval_count", 0))), 1)
        merged_rc = (int(float(nd_a.get("retrieval_count", 0)))
                     + int(float(nd_b.get("retrieval_count", 0))))
        merged_sc = (int(float(nd_a.get("success_count", 0)))
                     + int(float(nd_b.get("success_count", 0))))
        avg_a = float(nd_a.get("avg_steps", 0))
        avg_b = float(nd_b.get("avg_steps", 0))
        merged_avg = ((avg_a * rc_a + avg_b * rc_b) / (rc_a + rc_b)
                      if (avg_a > 0 or avg_b > 0) else 0.0)

        for k, v in merged_attrs.items():
            nd_a[k] = v
        nd_a["retrieval_count"] = merged_rc
        nd_a["success_count"] = merged_sc
        nd_a["avg_steps"] = merged_avg
        nd_a["utility"] = self._compute_utility_score(merged_sc, merged_rc, merged_avg)

        for he_id in list(self.get_node_hyperedges(node_b)):
            if not self.G.has_edge(he_id, node_a):
                self.G.add_edge(he_id, node_a, role="membership")

        self.G.remove_node(node_b)
        self.node_emb.remove(node_b)

        embed_parts = [merged_attrs.get("name", nd_a.get("name", ""))]
        content = merged_attrs.get("content", nd_a.get("content", ""))
        if content:
            embed_parts.append(str(content))
        full_text = ". ".join(embed_parts)

        self.node_emb.remove(node_a)
        new_vec = self.embed_texts([full_text])[0]
        self.node_emb.add(node_a, full_text, new_vec)

        self._dirty = True
        logger.info("  merge: '%s' + '%s' -> '%s' (type=%s)",
                    nd_a.get("name", ""), _nd_b_name_snapshot,
                    merged_attrs.get("name", nd_a.get("name", "")),
                    nd_a.get("type", ""))
        content_a = str(nd_a.get("content", "") or "").replace("\n", " ")
        content_b = _nd_b_content_snapshot.replace("\n", " ")
        logger.info("    merge_winner_content (A): %s", content_a[:500])
        logger.info("    merge_absorbed_content (B): %s", content_b[:500])

    def _structural_merge_abstract(self, threshold: float,
                                   propagation_depth: int,
                                   merge_alpha: float,
                                   llm_merge_fn=None) -> int:
        """Merge skill/mistake nodes via L-step structural propagation."""
        candidates = self._find_merge_candidates(
            threshold, propagation_depth, merge_alpha
        )
        consumed: Set[str] = set()
        merged_count = 0

        for nid_a, nid_b, _sim in candidates:
            if nid_a in consumed or nid_b in consumed:
                continue
            if not self.G.has_node(nid_a) or not self.G.has_node(nid_b):
                continue

            nd_a = dict(self.G.nodes[nid_a])
            nd_b = dict(self.G.nodes[nid_b])

            if llm_merge_fn:
                merged_attrs = llm_merge_fn(nd_a, nd_b)
            else:
                keep = (nd_a if float(nd_a.get("utility", 0))
                        >= float(nd_b.get("utility", 0)) else nd_b)
                skip = {"retrieval_count", "success_count", "role", "id",
                        "avg_steps", "utility"}
                merged_attrs = {k: v for k, v in keep.items() if k not in skip}

            self._merge_nodes(nid_a, nid_b, merged_attrs)
            consumed.update([nid_a, nid_b])
            merged_count += 1

        if merged_count > 0:
            self.node_emb.rewrite()
        return merged_count


    def stats(self) -> Dict[str, Any]:
        regular = [n for n in self.G.nodes if self.G.nodes[n].get("role") == "node"]
        he = [n for n in self.G.nodes if self.G.nodes[n].get("role") == "hyperedge"]
        utilities = [float(self.G.nodes[n].get("utility", 1.0)) for n in regular]
        avg_util = sum(utilities) / len(utilities) if utilities else 0.0
        return {
            "total_nodes": len(regular),
            "subtask_nodes": sum(1 for n in regular
                                 if self.G.nodes[n].get("type") == SUBTASK),
            "skill_nodes": sum(1 for n in regular
                               if self.G.nodes[n].get("type") == SKILL),
            "mistake_nodes": sum(1 for n in regular
                                 if self.G.nodes[n].get("type") == MISTAKE),
            "hyperedges": len(he),
            "graph_edges": self.G.number_of_edges(),
            "avg_utility": round(avg_util, 4),
            "t_min": (None if math.isinf(self._global_t_min)
                      else self._global_t_min),
            "t_max": self._global_t_max,
        }
