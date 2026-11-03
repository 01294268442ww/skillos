"""
Core types for the HyperSkill memory system
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
class MemoryStatus(Enum):
    """Status of memory operation"""
    BEGIN = "begin"
    IN = "in"
class MemoryType(Enum):
    """Type of memory framework"""
    HYPERSKILL = "hyperskill"
PROVIDER_MAPPING = {
    MemoryType.HYPERSKILL: ("HyperSkillProvider", "provider"),
}
class MemoryItemType(Enum):
    """Type of memory item content"""
    TEXT = "text"
    API = "api"
@dataclass
class MemoryRequest:
    """Request for memory retrieval"""
    query: str
    context: str
    status: MemoryStatus
    additional_params: Optional[Dict[str, Any]] = None
@dataclass
class MemoryItem:
    """Base memory item structure"""
    id: str
    content: Any
    metadata: Dict[str, Any]
    score: Optional[float] = None
    type: MemoryItemType = MemoryItemType.TEXT
@dataclass
class MemoryResponse:
    """Response containing retrieved memories"""
    memories: List[MemoryItem]
    memory_type: MemoryType
    total_count: int
    request_id: Optional[str] = None
@dataclass
class TrajectoryData:
    """Data structure for memory ingestion"""
    query: str
    trajectory: List[Dict[str, Any]]
    result: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None