"""
Agents Package
Contains all specialized AI units for the AgriBot framework.
All agents must inherit from BaseAgent.
"""

from .base import BaseAgent
from .soil import SoilIntelligenceAgent, MasterAgronomyAgent   # fixed: was AgronomyStrategyAgent
from .weather import MicroClimateAgent
from .vision import CropPathologyAgent

__all__ = [
    "BaseAgent",
    "SoilIntelligenceAgent",
    "MasterAgronomyAgent",      # canonical name from soil.py
    "MicroClimateAgent",
    "CropPathologyAgent",
]