"""
Utility functions for household energy data handling.
Simplified for incremental development.
"""

import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# Valid values for normalization
VALID_HOUSE_TYPES = [
    "Apartment", "Villa", "Bungalow", "Townhouse", "Penthouse"
]

VALID_CLIMATE_ZONES = [
    "Hot & Dry",
    "Hot & Humid",
    "Composite",
    "Temperate",
    "Cold"
]

VALID_CITY_TIERS = [
    "Tier 1",
    "Tier 2",
    "Tier 3"
]


def load_household_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load household energy requirement dataset."""
    if csv_path is None:
        base_dir = Path(__file__).parent.parent
        csv_path = os.path.join(base_dir, "household_energy_requirement.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    return pd.read_csv(csv_path)


def normalize_input(value: str, valid_values: List[str]) -> Optional[str]:
    """
    Normalize user input to match dataset values.
    Examples: "apt" -> "Apartment", "hot dry" -> "Hot & Dry"
    """
    if not value:
        return None
    
    value_lower = str(value).lower().strip()
    
    for valid in valid_values:
        if valid.lower() == value_lower or valid.lower().replace(" ", "") == value_lower.replace(" ", ""):
            return valid
    
    return None


def validate_input(profile: Dict) -> Tuple[bool, List[str]]:
    """
    Validate core household input fields.
    Returns: (is_valid, error_messages)
    """
    errors = []
    
    # Check required fields
    if not profile.get('house_type'):
        errors.append("house_type is required")
    elif not isinstance(profile['house_type'], str) or profile['house_type'] not in VALID_HOUSE_TYPES:
        errors.append(f"house_type must be one of: {', '.join(VALID_HOUSE_TYPES)}")
    
    if 'num_bedrooms' not in profile or profile['num_bedrooms'] is None:
        errors.append("num_bedrooms is required")
    elif not isinstance(profile['num_bedrooms'], (int, float)) or profile['num_bedrooms'] <= 0:
        errors.append("num_bedrooms must be > 0")
    
    if not profile.get('climate_zone'):
        errors.append("climate_zone is required")
    elif profile['climate_zone'] not in VALID_CLIMATE_ZONES:
        errors.append(f"climate_zone must be one of: {', '.join(VALID_CLIMATE_ZONES)}")
    
    if not profile.get('city_tier'):
        errors.append("city_tier is required")
    elif profile['city_tier'] not in VALID_CITY_TIERS:
        errors.append(f"city_tier must be one of: {', '.join(VALID_CITY_TIERS)}")
    
    if 'num_occupants' not in profile or profile['num_occupants'] is None:
        errors.append("num_occupants is required")
    elif not isinstance(profile['num_occupants'], (int, float)) or profile['num_occupants'] <= 0:
        errors.append("num_occupants must be > 0")
    
    return len(errors) == 0, errors


def find_similar_households(
    df: pd.DataFrame,
    profile: Dict,
    limit: int = 5
) -> List[Dict]:
    """
    Find similar households based on:
    - house_type
    - climate_zone
    - city_tier
    """
    filtered_df = df.copy()
    
    # Filter by house_type
    if profile.get('house_type'):
        filtered_df = filtered_df[filtered_df['house_type'] == profile['house_type']]
    
    # Filter by climate_zone
    if profile.get('climate_zone'):
        filtered_df = filtered_df[filtered_df['climate_zone'] == profile['climate_zone']]
    
    # Filter by city_tier
    if profile.get('city_tier'):
        filtered_df = filtered_df[filtered_df['city_tier'] == profile['city_tier']]
    
    # Convert to list of dicts
    similar = filtered_df.head(limit).to_dict('records')
    
    # Convert numpy types to Python types
    for record in similar:
        for key, value in record.items():
            if hasattr(value, 'item'):
                record[key] = value.item()
    
    return similar
