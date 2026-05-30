"""
Main entry point for Household Energy Requirement Calculator & Advisor.
Uses LangGraph for multi-agent workflow with state management.
"""

import json
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from langgraph.graph import StateGraph, START, END
from state import HouseholdProfileState
from agents.collect_household_profile_agent import CollectHouseholdProfileAgent

# Load environment variables (for OpenAI API key)
load_dotenv()

# Initialize agents
collect_household_agent = CollectHouseholdProfileAgent()


def collect_household_profile_node(state: HouseholdProfileState) -> HouseholdProfileState:
    """LangGraph node: Collect household profile."""
    return collect_household_agent.process(state)


def build_workflow():
    """Build the LangGraph workflow."""
    
    # Create the state graph
    workflow = StateGraph(HouseholdProfileState)
    
    # Add nodes
    workflow.add_node("collect_household_profile", collect_household_profile_node)
    
    # Add edges
    workflow.add_edge(START, "collect_household_profile")
    workflow.add_edge("collect_household_profile", END)
    
    # Compile the graph
    graph = workflow.compile()
    
    return graph


def run_with_custom_profile(profile_data: dict):
    """Run workflow with custom household profile."""
    print("\n" + "="*60)
    print("Agent 1: Collecting & Validating Household Profile")
    print("="*60 + "\n")
    
    # Initialize state with custom profile
    initial_state: HouseholdProfileState = {
        **profile_data,
        'messages': [],
        'errors': [],
        'workflow_stage': 'initialized',
    }
    
    # Build and run workflow
    graph = build_workflow()
    final_state = graph.invoke(initial_state)
    
    # Display results
    _display_results(final_state)
    
    return final_state


def _display_results(state: HouseholdProfileState):
    """Display workflow results."""
    
    print("\n[PROFILE COLLECTION RESULTS]")
    print("-" * 60)
    
    summary = collect_household_agent.get_summary(state)
    
    # Display profile
    print("\n[COLLECTED PROFILE]")
    for key, value in summary['profile'].items():
        if value is not None:
            print("   {}: {}".format(key.replace('_', ' ').title(), value))
    
    # Display validation status
    print("\n[VALIDATION STATUS]")
    stage = summary['stage']
    if stage == "profile_validated":
        print("   Status: [PASSED]")
    else:
        print("   Status: [FAILED]")
    
    # Display analysis
    print("\n[INITIAL ANALYSIS]")
    print("   Similar households found: {}".format(summary['analysis']['similar_households_found']))
    
    # Display similar households
    similar = state.get('similar_households', [])
    if similar:
        print("\n[SIMILAR HOUSEHOLDS SAMPLES]")
        for i, h in enumerate(similar[:3], 1):
            print("   {}. {} - {} bedrooms, {} occupants, {:.1f} kWh/day, ${:.0f}/month".format(
                i,
                h.get('house_type', 'N/A'),
                h.get('num_bedrooms', 'N/A'),
                h.get('num_occupants', 'N/A'),
                h.get('daily_energy_consumption_kWh', 0),
                h.get('monthly_energy_consumption_kWh', 0) * 8  # Assuming ₹8/kWh
            ))
    
    # Display messages from agent
    if summary['messages']:
        print("\n[AGENT MESSAGES]")
        for msg in summary['messages']:
            print("   {}".format(msg))
    
    # Display errors if any
    if summary['errors']:
        print("\n[VALIDATION ERRORS]")
        for error in summary['errors']:
            print("   - {}".format(error))
    
    print("\n" + "="*60)


def main():
    """Main entry point."""
    
    print("\n[AGENT 1: COLLECT & VALIDATE HOUSEHOLD PROFILE]")
    print("=" * 60)
    print("\nSimplified, focused implementation:")
    print("- Collect core inputs (house_type, bedrooms, climate, tier, occupants)")
    print("- Normalize to dataset format")
    print("- Validate against constraints")
    print("- Find similar households for benchmarking\n")
    
    # Example 1: Valid household - Apartment in Hot & Dry
    print("[EXAMPLE 1]: Valid apartment profile")
    profile_1 = {
        'house_type': 'Apartment',
        'num_bedrooms': 2,
        'climate_zone': 'Hot & Dry',
        'city_tier': 'Tier 3',
        'num_occupants': 3,
    }
    state_1 = run_with_custom_profile(profile_1)
    
    # Example 2: Valid household - Villa in Hot & Humid
    print("\n\n[EXAMPLE 2]: Valid villa profile")
    profile_2 = {
        'house_type': 'Villa',
        'num_bedrooms': 3,
        'climate_zone': 'Hot & Humid',
        'city_tier': 'Tier 2',
        'num_occupants': 5,
    }
    state_2 = run_with_custom_profile(profile_2)
    
    # Example 3: Input with normalization - user enters "apt" instead of "Apartment"
    print("\n\n[EXAMPLE 3]: Profile with input normalization")
    profile_3 = {
        'house_type': 'apt',  # Will be normalized to "Apartment"
        'num_bedrooms': 1,
        'climate_zone': 'cold',  # Will be normalized to "Cold"
        'city_tier': 'tier 1',  # Will be normalized to "Tier 1"
        'num_occupants': 2,
    }
    state_3 = run_with_custom_profile(profile_3)
    
    # Example 4: Invalid household - missing required field
    print("\n\n[EXAMPLE 4]: Invalid profile (missing climate_zone)")
    profile_4 = {
        'house_type': 'Bungalow',
        'num_bedrooms': 4,
        'city_tier': 'Tier 2',
        'num_occupants': 6,
        # Missing climate_zone
    }
    state_4 = run_with_custom_profile(profile_4)
    
    print("\n\n[WORKFLOW SUMMARY]")
    print("="*60)
    print("Agent 1 is now simplified and focused.")
    print("Next: Add Agent 2 (Capture Occupancy Details) when ready.")



if __name__ == "__main__":
    main()