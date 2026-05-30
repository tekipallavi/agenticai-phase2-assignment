"""
State management for Household Energy Advisor workflow.
Maintains state across all agent nodes.
Simplified for incremental approach.
"""

from typing import TypedDict, Optional, List, Dict, Any


class HouseholdProfileState(TypedDict, total=False):
    """State object maintained across all agent nodes."""
    
    # Agent 1: Core Inputs - Only what we need now
    house_type: Optional[str]
    num_bedrooms: Optional[int]
    floor_area_sqft: Optional[float]
    climate_zone: Optional[str]
    city_tier: Optional[str]
    num_occupants: Optional[int]
    
    # Metadata
    messages: List[str]
    errors: List[str]
    current_agent: Optional[str]
    workflow_stage: str
    
    # Agent 1 Analysis Results
    similar_households: Optional[List[Dict[str, Any]]]
    similar_households_count: Optional[int]
