# -*- coding: utf-8 -*-
"""HERA subsystem: analysis methods + plan making (scientific brain).

V2.1 adds the outer research loop authority: MethodPortfolio (candidate
branches + mutation axes) and Prioritizer (PrioritizationTicket + frozen
ResearchProgramGrant + SnapshotReadyV3). HERA never executes or evaluates.
"""
from hera.analyzer import Analyzer
from hera.interpreter import Interpreter, Interpretation
from hera.memory import ScientificMemory
from hera.planner import Planner
from hera.portfolio import (MethodPortfolio, PortfolioBranch,
                            ResourceProfiler)
from hera.prioritization import Prioritizer

__all__ = [
    "Analyzer", "Planner", "Interpreter", "Interpretation", "ScientificMemory",
    "MethodPortfolio", "PortfolioBranch", "ResourceProfiler", "Prioritizer",
]
