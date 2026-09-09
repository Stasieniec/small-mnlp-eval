"""Analyses that compare systems to each other rather than to a test set."""

from mnlp_eval.analysis.overlap import OverlapResult, compare_subnetworks, overlap_report
from mnlp_eval.analysis.subnetwork import Subnetwork, load_subnetwork, load_subnetworks

__all__ = [
    "OverlapResult",
    "Subnetwork",
    "compare_subnetworks",
    "load_subnetwork",
    "load_subnetworks",
    "overlap_report",
]
