"""HyperSkill memory provider."""

import os
import json
import math
import uuid
import logging
from typing import List, Optional, Dict, Any, Tuple

from .base_memory import BaseMemoryProvider
from .memory_types import (
    MemoryRequest,
    MemoryResponse,
    TrajectoryData,
    MemoryType,
    MemoryItem,
    MemoryStatus,
)
from .hypergraph_core import (
    HyperGraph,
    load_embedding_model,
    SUBTASK,
    SKILL,
    MISTAKE,
    EPISODE,
)
from .prompts import (
    get_initial_plan_prompt,
    get_success_extraction_prompt,
    get_failure_extraction_prompt,
    get_merge_prompt,
    EXPERIENCE_HEADER,
    SKILL_HEADER,
    MISTAKE_HEADER,
)

logger = logging.getLogger(__name__)
class HyperSkillProvider(BaseMemoryProvider):
    """HyperSkill clean hypergraph memory provider."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(MemoryType.HYPERSKILL, config)

        self.storage_dir = self.config.get("storage_dir", "./storage/hyperskill")

        self.k = int(os.environ.get("HYPERSKILL_K", self.config.get("k", 5)))

        self.beta = float(os.environ.get("HYPERSKILL_BETA", self.config.get("beta", 0.7)))

        self.maintenance_ratio = float(self.config.get("maintenance_ratio", 0.1))
        mr_env = os.environ.get("HYPERSKILL_MAINTENANCE_RATIO")
        if mr_env is not None:
            self.maintenance_ratio = float(mr_env)

        self.prune_min_retrievals = int(self.config.get("prune_min_retrievals", 3))
        self.prune_threshold = float(
            os.environ.get("HYPERSKILL_PRUNE_THRESHOLD",
                           self.config.get("prune_threshold", 0.2))
        )
        self.merge_threshold = float(
            os.environ.get("HYPERSKILL_MERGE_THRESHOLD",
                           self.config.get("merge_threshold", 0.85))
        )
        self.merge_propagation_depth = int(self.config.get("merge_propagation_depth", 2))
        self.merge_alpha = float(self.config.get("merge_alpha", 0.5))

        self.skill_dedup_threshold = float(
            self.config.get("skill_semantic_dedup_threshold", 0.90)
        )

        self.embedding_model_name = self.config.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_model_cache = self.config.get(
            "embedding_model_cache", "./storage/models"
        )

        self.domain = os.environ.get("HYPERSKILL_DOMAIN", self.config.get("domain", "general"))

        self.model = self.config.get("model")

        self._graph: Optional[HyperGraph] = None
        self._embedding_model = None
        self._retrieved_node_ids: List[str] = []
        self._cached_plan: Optional[List[dict]] = None
        self._cached_begin_memory: Optional[str] = None
        self._last_pipeline_meta: Optional[dict] = None
        self._total_tasks = int(self.config.get("total_tasks", 0))


    def initialize(self) -> bool:
        try:
            abs_storage = os.path.abspath(self.storage_dir)
            abs_cwd = os.path.abspath(".")
            if not abs_storage.startswith(abs_cwd):
                logger.warning(
                    "storage_dir '%s' resolves outside project root '%s'. "
                    "Check STORAGE_PREFIX setting.",
                    abs_storage, abs_cwd,
                )
            os.makedirs(self.storage_dir, exist_ok=True)

            self._embedding_model = load_embedding_model(
                model_name=self.embedding_model_name,
                cache_dir=self.embedding_model_cache,
            )

            graph_config = {
                "beta": self.beta,
                "skill_semantic_dedup_threshold": self.skill_dedup_threshold,
            }
            self._graph = HyperGraph(
                base_dir=self.storage_dir,
                embedding_model=self._embedding_model,
                config=graph_config,
            )

            stats = self._graph.stats()
            logger.info(
                "HyperSkill initialized: %d nodes (%d st, %d sk, %d mk), %d HEs",
                stats["total_nodes"],
                stats["subtask_nodes"], stats["skill_nodes"], stats["mistake_nodes"],
                stats["hyperedges"],
            )
            return True
        except Exception as e:
            logger.error("Failed to initialize HyperSkill %s", e, exc_info=True)
            return False

    def get_last_pipeline_meta(self) -> Optional[dict]:
        """Return pipeline metadata from the last provide_memory(BEGIN) call."""
        return self._last_pipeline_meta


    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        memories: List[MemoryItem] = []

        if self._graph is None:
            return MemoryResponse(
                memories=[], memory_type=self.memory_type, total_count=0
            )

        try:
            query = request.query or ""

            if request.status != MemoryStatus.BEGIN:
                return MemoryResponse(
                    memories=[], memory_type=self.memory_type, total_count=0
                )

            graph_stats = self._graph.stats()

            plan_subtasks = self._parse_plan(request)
            if plan_subtasks is None and self.model:
                plan_subtasks = self._generate_initial_plan(query, domain=self.domain)
            if plan_subtasks:
                logger.info("[%s] decomposition (%d steps):", self.domain, len(plan_subtasks))
                for i, s in enumerate(plan_subtasks, 1):
                    logger.info("  step %d: %s - %s",
                                i, s.get("name", "?"),
                                (s.get("description", "") or "")[:100])
            else:
                logger.info("[%s]: (no decomposition)", self.domain)

            self._cached_plan = plan_subtasks

            result = self._graph.retrieve(
                task_desc=query,
                plan_subtasks=plan_subtasks,
                k=self.k,
            )

            self._retrieved_node_ids = result.get("retrieved_node_ids", [])

            skills = result.get("skills", [])
            mistakes = result.get("mistakes", [])
            hyperedges = result.get("hyperedges", [])
            debug = result.get("debug", {})

            logger.info(
                "retrieve: %d HEs, %d skills, %d mistakes "
                "(|E_traj|=%d |E_sub|=%d |E*|=%d |V_sub|=%d |V_sm|=%d) "
                "[graph: %d nodes, %d HEs]",
                len(hyperedges), len(skills), len(mistakes),
                len(debug.get("e_task", [])),
                len(debug.get("e_sub", [])),
                len(debug.get("all_he_ids", [])),
                len(debug.get("v_sub", [])),
                len(debug.get("v_sm", [])),
                graph_stats["total_nodes"], graph_stats["hyperedges"],
            )
            if skills or mistakes:
                sk_info = [
                    f"{s.get('name', '?')}(sim={s.get('_sim', 0):.2f}|cooc={s.get('_co_occurrence', 0)})"
                    for s in skills
                ]
                mk_info = [
                    f"{m.get('name', '?')}(sim={m.get('_sim', 0):.2f}|cooc={m.get('_co_occurrence', 0)})"
                    for m in mistakes
                ]
                logger.info("top skills: [%s]", ", ".join(sk_info))
                logger.info("top mistakes: [%s]", ", ".join(mk_info))

            content = self._format_memory(hyperedges, skills, mistakes)
            self._cached_begin_memory = content

            def _node_detail(n: dict) -> dict:
                # _rank is the sort key (kappa, sim): co-occurrence count first,
                # similarity to the task as the tie-break.
                rank = [round(float(x), 4) for x in n.get("_rank", (0, 0))]
                return {
                    "name": n.get("name", ""),
                    "content": (n.get("content", "") or "")[:300],
                    "sim": round(float(n.get("_sim", 0)), 4),
                    "cooc": int(n.get("_co_occurrence", 0)),
                    "rank": rank,
                    "utility": float(n.get("utility", 0.5)),
                    "retrieval_count": int(float(n.get("retrieval_count", 0))),
                }

            pipeline_meta = {
                "type": "hyperskill",
                "status": "begin",
                "graph_nodes": graph_stats["total_nodes"],
                "graph_subtasks": graph_stats["subtask_nodes"],
                "graph_skills": graph_stats["skill_nodes"],
                "graph_mistakes": graph_stats["mistake_nodes"],
                "graph_hyperedges": graph_stats["hyperedges"],
                "initial_plan": [
                    {"name": s.get("name", ""), "description": s.get("description", "")}
                    for s in (plan_subtasks or [])
                ],
                "e_traj_count": len(debug.get("e_task", [])),
                "e_sub_count": len(debug.get("e_sub", [])),
                "all_he_count": len(debug.get("all_he_ids", [])),
                "v_sub_count": len(debug.get("v_sub", [])),
                "v_sm_count": len(debug.get("v_sm", [])),
                "retrieved_hyperedges": [
                    {
                        "id": h.get("id", ""),
                        "fragment": (h.get("fragment", "") or "")[:250],
                        "status": h.get("status", ""),
                        "sim": round(float(h.get("_sim", 0)), 4),
                        "utility": float(h.get("utility", 0.5)),
                    }
                    for h in hyperedges
                ],
                "retrieved_skills": [_node_detail(s) for s in skills],
                "retrieved_mistakes": [_node_detail(m) for m in mistakes],
            }
            self._last_pipeline_meta = pipeline_meta

            memories.append(MemoryItem(
                id=f"hs_begin_{uuid.uuid4().hex[:8]}",
                content=content or "",
                metadata=pipeline_meta,
                score=1.0,
            ))

            return MemoryResponse(
                memories=memories,
                memory_type=self.memory_type,
                total_count=len(memories),
                request_id=str(uuid.uuid4()),
            )

        except Exception as e:
            logger.error("Error in provide_memory: %s", e, exc_info=True)
            return MemoryResponse(
                memories=[], memory_type=self.memory_type, total_count=0
            )


    def take_in_memory(self, trajectory_data: TrajectoryData) -> Tuple[bool, str]:
        if self._graph is None:
            return False, "Graph not initialized"

        try:
            metadata = trajectory_data.metadata or {}
            is_correct = metadata.get("is_correct", False)
            step_count = metadata.get("step_count", metadata.get("steps", 0))
            if step_count == 0:
                step_count = len(trajectory_data.trajectory) if trajectory_data.trajectory else 1

            outcome = "success" if is_correct else "failure"
            extracted = self._extract_with_llm(trajectory_data, is_correct)
            if not extracted:
                logger.warning("extract: LLM extraction failed for %s", outcome)
                return False, "LLM extraction failed"

            sk_names = [s.get("name", "?") for s in extracted.get("skills", [])]
            mk_names = [m.get("name", "?") for m in extracted.get("mistakes", [])]
            logger.info("extract (%s): skills=[%s] mistakes=[%s]",
                        outcome, ", ".join(sk_names), ", ".join(mk_names))
            frag = extracted.get("fragment", "") or ""
            if frag:
                logger.info("  fragment: %s", frag[:200])

            absorbed = []

            subtask_ids = []
            plan_steps = self._cached_plan or []
            for st in plan_steps:
                name = st.get("name", st.get("action_name", ""))
                content = st.get("description", st.get("content", ""))
                if not name:
                    continue
                nid = self._graph.add_subtask(name=name, content=content)
                if nid:
                    subtask_ids.append(nid)

            skill_ids = []
            for sk in extracted.get("skills", []):
                name = sk.get("name", "")
                content = sk.get("content", "")
                if not name:
                    continue
                nid = self._graph.add_skill(
                    name=name, content=content, metadata=sk.get("metadata"),
                )
                if nid:
                    skill_ids.append(nid)

            mistake_ids = []
            for mk in extracted.get("mistakes", []):
                name = mk.get("name", "")
                content = mk.get("content", "")
                if not name:
                    continue
                nid = self._graph.add_mistake(
                    name=name, content=content, metadata=mk.get("metadata"),
                )
                if nid:
                    mistake_ids.append(nid)

            absorbed.append(
                f"Added {len(subtask_ids)} subtasks, "
                f"{len(skill_ids)} skills, {len(mistake_ids)} mistakes"
            )

            status_label = "success" if is_correct else "failure"
            task_type = metadata.get("task_type", "")
            he_meta = {"status": status_label}
            if task_type:
                he_meta["task_type"] = task_type

            episode_node_ids = subtask_ids + skill_ids + mistake_ids
            if episode_node_ids:
                self._graph.add_hyperedge(
                    edge_type=EPISODE,
                    node_ids=episode_node_ids,
                    fragment=frag or f"Episode ({status_label})",
                    task_query=trajectory_data.query,
                    metadata=he_meta,
                )

            if self._retrieved_node_ids:
                self._graph.update_utility(
                    self._retrieved_node_ids, is_correct, step_count
                )
                absorbed.append(
                    f"Updated gamma for {len(self._retrieved_node_ids)} retrieved nodes"
                )

            self._retrieved_node_ids = []
            self._cached_begin_memory = None
            self._cached_plan = None

            counter_file = os.path.join(self.storage_dir, "episode_count.txt")
            try:
                ep_count = int(open(counter_file).read().strip()) if os.path.exists(counter_file) else 0
            except (ValueError, IOError):
                ep_count = 0
            ep_count += 1
            with open(counter_file, "w") as f:
                f.write(str(ep_count))

            if self.maintenance_ratio > 0:
                if self._total_tasks > 0:
                    interval = max(1, math.ceil(self.maintenance_ratio * self._total_tasks))
                else:
                    interval = max(1, int(1.0 / self.maintenance_ratio))
                if ep_count % interval == 0:
                    logger.info("maintenance at episode %d (interval=%d)", ep_count, interval)
                    maint = self._graph.maintain_graph(
                        merge_threshold=self.merge_threshold,
                        propagation_depth=self.merge_propagation_depth,
                        merge_alpha=self.merge_alpha,
                        prune_min_retrievals=self.prune_min_retrievals,
                        prune_threshold=self.prune_threshold,
                        llm_merge_fn=self._llm_merge_nodes,
                    )
                    absorbed.append(
                        f"Maintenance: pruned={maint['pruned']}, "
                        f"merged={maint['merged']}, orphans={maint['orphans']}"
                    )

            self._graph.flush()

            return True, " | ".join(absorbed)

        except Exception as e:
            logger.error("Error in take_in_memory: %s", e, exc_info=True)
            return False, f"Error: {e}"


    def _format_memory(self, hyperedges: List[dict],
                       skills: List[dict],
                       mistakes: List[dict]) -> Optional[str]:
        """Format retrieved memory for the agent prompt.

        Sections:
        """
        parts = []

        if hyperedges:
            lines = [EXPERIENCE_HEADER]
            for i, he in enumerate(hyperedges, 1):
                fragment = (he.get("fragment", "") or "").strip()
                status = he.get("status", "")
                if not fragment:
                    continue
                if status:
                    lines.append(f"{i}. [{status}] {fragment}")
                else:
                    lines.append(f"{i}. {fragment}")
            if len(lines) > 1:  # at least one fragment was non-empty
                parts.append("\n".join(lines))

        if skills:
            lines = [SKILL_HEADER]
            for i, sk in enumerate(skills, 1):
                name = sk.get("name", "")
                content = sk.get("content", "")
                lines.append(f"{i}. **{name}**: {content}")
            parts.append("\n".join(lines))

        if mistakes:
            lines = [MISTAKE_HEADER]
            for i, mk in enumerate(mistakes, 1):
                name = mk.get("name", "")
                content = mk.get("content", "")
                lines.append(f"{i}. **{name}**: {content}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts) if parts else None


    def _extract_with_llm(self, trajectory_data: TrajectoryData,
                          is_correct: bool) -> Optional[Dict[str, Any]]:
        if not self.model:
            return self._fallback_extraction(trajectory_data, is_correct)

        try:
            traj_text = self._format_trajectory(trajectory_data)
            outcome = "CORRECT" if is_correct else "INCORRECT"
            prompt_template = (
                get_success_extraction_prompt(self.domain) if is_correct
                else get_failure_extraction_prompt(self.domain)
            )
            prompt = prompt_template.format(
                query=trajectory_data.query,
                trajectory=traj_text,
                result=trajectory_data.result or (
                    "Completed successfully" if is_correct else "Failed or incorrect"
                ),
                outcome=outcome,
            )
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            response = self.model(messages)
            content = getattr(response, "content", str(response))
            return self._parse_extraction(content)

        except Exception as e:
            logger.error("LLM extraction failed: %s", e)
            return self._fallback_extraction(trajectory_data, is_correct)

    def _parse_extraction(self, content: str) -> Optional[Dict]:
        """Parse LLM JSON response.

        schema:
            {"skills": [{"name", "content"}], "knowledge_fragment": "..."}
          or
            {"mistakes": [{"name", "content"}], "knowledge_fragment": "..."}
        """
        content = content.strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            content = content[start:end]

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("failed to parse LLM extraction as JSON")
            return None

        fragment = data.get("knowledge_fragment", data.get("fragment", ""))

        result = {
            "skills": [],
            "mistakes": [],
            "fragment": fragment,
        }

        for item in data.get("skills", []):
            if not isinstance(item, dict):
                continue
            name = item.get("name", item.get("title", ""))
            content = item.get("content", item.get("principle", ""))
            if not name:
                continue
            result["skills"].append({
                "name": str(name),
                "content": str(content),
                "metadata": {
                    k: str(item[k]) for k in ("when_to_apply", "why_it_works")
                    if k in item and item[k]
                },
            })

        for item in data.get("mistakes", []):
            if not isinstance(item, dict):
                continue
            name = item.get("name", item.get("title", ""))
            content = item.get("content", item.get("description", ""))
            if not name:
                continue
            result["mistakes"].append({
                "name": str(name),
                "content": str(content),
                "metadata": {
                    k: str(item[k]) for k in ("why_it_happens", "how_to_avoid")
                    if k in item and item[k]
                },
            })

        return result

    def _llm_merge_nodes(self, nd_a: dict, nd_b: dict) -> dict:
        """Use LLM to consolidate two skills/mistakes into one higher-order node."""
        if not self.model:
            keep = nd_a if float(nd_a.get("utility", 0)) >= float(nd_b.get("utility", 0)) else nd_b
            skip = {"retrieval_count", "success_count", "role", "id", "avg_steps", "utility"}
            return {k: v for k, v in keep.items() if k not in skip}

        ntype = nd_a.get("type", "skill")
        name_a, content_a = nd_a.get("name", ""), nd_a.get("content", "")
        name_b, content_b = nd_b.get("name", ""), nd_b.get("content", "")

        prompt = get_merge_prompt(ntype, name_a, content_a, name_b, content_b)

        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            response = self.model(messages)
            raw = getattr(response, "content", str(response)).strip()

            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]

            data = json.loads(raw)
            skip = {"retrieval_count", "success_count", "role", "id", "avg_steps", "utility"}
            result = {k: v for k, v in nd_a.items() if k not in skip}
            result["name"] = str(data.get("name", name_a))
            result["content"] = str(data.get("content", content_a))

            logger.info("  LLM merge: '%s' + '%s' -> '%s'", name_a, name_b, result["name"])
            return result

        except Exception as e:
            logger.warning("LLM merge failed: %s, falling back", e)
            keep = nd_a if float(nd_a.get("utility", 0)) >= float(nd_b.get("utility", 0)) else nd_b
            skip = {"retrieval_count", "success_count", "role", "id", "avg_steps", "utility"}
            return {k: v for k, v in keep.items() if k not in skip}

    def _fallback_extraction(self, trajectory_data: TrajectoryData,
                             is_correct: bool) -> Dict[str, Any]:
        query = trajectory_data.query or ""
        return {
            "skills": [],
            "mistakes": [],
            "fragment": f"{'Success' if is_correct else 'Failure'}: {query[:80]}",
        }


    def _generate_initial_plan(self, query: str,
                               domain: str = "general") -> Optional[List[dict]]:
        if not self.model:
            return None
        try:
            prompt_template = get_initial_plan_prompt(domain)
            prompt = prompt_template.format(query=query)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            response = self.model(messages)
            content = getattr(response, "content", str(response))
            return self._parse_plan_json(content)
        except Exception as e:
            logger.warning("initial plan generation failed: %s", e)
            return None

    def _parse_plan_json(self, content: str) -> Optional[List[dict]]:
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            content = content[start:end]

        try:
            data = json.loads(content)
            if isinstance(data, list):
                return [
                    {"name": str(s.get("name", "")),
                     "description": str(s.get("description", ""))}
                    for s in data if isinstance(s, dict)
                ]
        except json.JSONDecodeError:
            pass
        return None

    def _parse_plan(self, request: MemoryRequest) -> Optional[List[dict]]:
        """Extract plan subtasks from request.additional_params only."""
        params = request.additional_params or {}
        plan = params.get("plan_subtasks") or params.get("plan")
        if plan and isinstance(plan, list):
            return plan
        return None

    def _format_trajectory(self, trajectory_data: TrajectoryData) -> str:
        if not trajectory_data.trajectory:
            return "No execution trajectory available"

        parts = [f"Task: {trajectory_data.query}", ""]
        for i, step in enumerate(trajectory_data.trajectory, 1):
            step_type = step.get("type", "step")
            content = step.get("content", "")
            parts.append(f"Step {i} ({step_type}): {content}")

        if trajectory_data.result:
            parts.append(f"\nFinal Result: {trajectory_data.result}")

        return "\n".join(parts)
