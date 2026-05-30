"""
Agent 1: Collect & Validate Household Profile
Lightweight, focused implementation.
Core responsibilities:
1. Collect & Validate core inputs
2. Normalize inputs to match dataset
3. Basic validation
4. Load dataset
5. Find similar households
"""

import sys
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from state import HouseholdProfileState
from utils import (
    load_household_data,
    normalize_input,
    validate_input,
    find_similar_households,
    VALID_HOUSE_TYPES,
    VALID_CLIMATE_ZONES,
    VALID_CITY_TIERS
)


class CollectHouseholdProfileAgent:
    """
    Agent 1: Collect & Validate Household Profile
    
    Responsibilities:
    ✓ Collect core inputs
    ✓ Normalize inputs
    ✓ Validate inputs
    ✓ Load dataset
    ✓ Find similar households for initial analysis
    """
    
    def __init__(self):
        self.agent_name = "Agent1_CollectProfile"
        self.df = load_household_data()
    
    def process(self, state: HouseholdProfileState) -> HouseholdProfileState:
        """Process household profile collection and basic analysis."""
        
        state['current_agent'] = self.agent_name
        state['workflow_stage'] = "collecting_profile"
        state['messages'] = state.get('messages', [])
        state['errors'] = state.get('errors', [])
        
        try:
            # Step 1: Normalize inputs
            state = self._normalize_inputs(state)
            
            # Step 2: Validate inputs
            is_valid = self._validate_inputs(state)
            if not is_valid:
                state['workflow_stage'] = "validation_failed"
                return state
            
            # Step 3: Find similar households (lightweight analysis)
            state = self._find_similar_households(state)
            
            state['messages'].append(
                "[SUCCESS] Profile collected and validated for {} ({} bedrooms, {})".format(
                    state.get('house_type'),
                    state.get('num_bedrooms'),
                    state.get('climate_zone')
                )
            )
            state['workflow_stage'] = "profile_validated"
            
        except Exception as e:
            state['errors'].append(str(e))
            state['messages'].append("[ERROR] {}".format(str(e)))
        
        return state
    
    def _normalize_inputs(self, state: HouseholdProfileState) -> HouseholdProfileState:
        """Normalize user inputs to match dataset format."""
        
        state['messages'].append("[STEP 1] Normalizing inputs...")
        
        # Normalize house_type
        if state.get('house_type'):
            normalized = normalize_input(state['house_type'], VALID_HOUSE_TYPES)
            if normalized:
                state['house_type'] = normalized
                state['messages'].append("  -> house_type: {} -> {}".format(
                    state.get('house_type'), normalized
                ))
        
        # Normalize climate_zone
        if state.get('climate_zone'):
            normalized = normalize_input(state['climate_zone'], VALID_CLIMATE_ZONES)
            if normalized:
                state['climate_zone'] = normalized
                state['messages'].append("  -> climate_zone: normalized to {}".format(normalized))
        
        # Normalize city_tier
        if state.get('city_tier'):
            normalized = normalize_input(state['city_tier'], VALID_CITY_TIERS)
            if normalized:
                state['city_tier'] = normalized
                state['messages'].append("  -> city_tier: normalized to {}".format(normalized))
        
        return state
    
    def _validate_inputs(self, state: HouseholdProfileState) -> bool:
        """Validate core inputs."""
        
        state['messages'].append("[STEP 2] Validating inputs...")
        
        # Create validation profile
        profile = {
            'house_type': state.get('house_type'),
            'num_bedrooms': state.get('num_bedrooms'),
            'climate_zone': state.get('climate_zone'),
            'city_tier': state.get('city_tier'),
            'num_occupants': state.get('num_occupants'),
        }
        
        is_valid, errors = validate_input(profile)
        
        if is_valid:
            state['messages'].append("  -> All validations passed!")
            return True
        else:
            state['errors'].extend(errors)
            for error in errors:
                state['messages'].append("  -> VALIDATION ERROR: {}".format(error))
            return False
    
    def _find_similar_households(self, state: HouseholdProfileState) -> HouseholdProfileState:
        """Find similar households from dataset."""
        
        state['messages'].append("[STEP 3] Finding similar households...")
        
        profile = {
            'house_type': state.get('house_type'),
            'climate_zone': state.get('climate_zone'),
            'city_tier': state.get('city_tier'),
        }
        
        similar = find_similar_households(self.df, profile, limit=5)
        
        state['similar_households'] = similar
        state['similar_households_count'] = len(similar)
        
        state['messages'].append("  -> Found {} similar households".format(len(similar)))
        
        # Show sample data
        if similar:
            sample = similar[0]
            state['messages'].append("  -> Sample: {} ({} bedr, {} occ, {} kWh/day)".format(
                sample.get('house_type', 'N/A'),
                sample.get('num_bedrooms', 'N/A'),
                sample.get('num_occupants', 'N/A'),
                round(sample.get('daily_energy_consumption_kWh', 0), 2)
            ))
        
        return state
    
    def get_summary(self, state: HouseholdProfileState) -> Dict[str, Any]:
        """Get summary of collected profile."""
        return {
            'agent': self.agent_name,
            'stage': state.get('workflow_stage'),
            'profile': {
                'house_type': state.get('house_type'),
                'bedrooms': state.get('num_bedrooms'),
                'climate_zone': state.get('climate_zone'),
                'city_tier': state.get('city_tier'),
                'occupants': state.get('num_occupants'),
                'floor_area': state.get('floor_area_sqft'),
            },
            'analysis': {
                'similar_households_found': state.get('similar_households_count', 0),
            },
            'messages': state.get('messages', []),
            'errors': state.get('errors', []),
        }
