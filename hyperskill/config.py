"""Default configuration for the HyperSkill memory system."""

import os
from typing import Any, Dict

from .memory_types import MemoryType

STORAGE_BASE_DIR = "./storage"

DEFAULT_CONFIG = {
    "providers": {
        MemoryType.HYPERSKILL: {
            "storage_dir": os.path.join(STORAGE_BASE_DIR, "hyperskill"),
            # retrieval budget k (Sec. 3.3-3.4)
            "k": 5,
            # utility score gamma (Eq. 3)
            "beta": 0.7,
            # maintenance (Sec. 3.5): run every ceil(ratio * total_tasks) episodes
            "maintenance_ratio": 0.1,
            "prune_min_retrievals": 3,
            "prune_threshold": 0.2,
            "merge_threshold": 0.85,
            "merge_propagation_depth": 2,
            "merge_alpha": 0.5,
            # insertion-time semantic dedup for skill / mistake nodes
            "skill_semantic_dedup_threshold": 0.90,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_model_cache": os.path.join(STORAGE_BASE_DIR, "models"),
        },
    },
}


def get_memory_config(provider_type: MemoryType,
                      storage_prefix: str = None) -> Dict[str, Any]:
    """Return the configuration for a memory provider.

    Args:
        provider_type: which provider to configure.
        storage_prefix: optional prefix that isolates storage per model/benchmark,
            e.g. "qwen3-30b/gaia" makes paths "./storage/qwen3-30b/gaia/...".
            Falls back to the STORAGE_PREFIX environment variable.
    """
    config = DEFAULT_CONFIG["providers"].get(provider_type, {}).copy()
    prefix = storage_prefix or os.environ.get("STORAGE_PREFIX")
    if prefix:
        prefix_dir = os.path.join(STORAGE_BASE_DIR, prefix)
        for key, val in config.items():
            if isinstance(val, str) and val.startswith(STORAGE_BASE_DIR):
                config[key] = val.replace(STORAGE_BASE_DIR, prefix_dir, 1)
    return config
