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
    return collect_household_agent.process(state)


def build_workflow():
   
    workflow = StateGraph(HouseholdProfileState)
    
    # Add nodes
    workflow.add_node("collect_household_profile", collect_household_profile_node)
    
    # Add edges
    workflow.add_edge(START, "collect_household_profile")
    workflow.add_edge("collect_household_profile", END)
    
    # Compile the graph
    graph = workflow.compile()
    
    return graph


def initializeProfile(profile_data: dict):
    initial_state: HouseholdProfileState = {
        "profile_data" : profile_data,
        "messages": [],
        "errors": [],
        "current_agent": '',
        "workflow_stage": 'Start'
    }
    
    # Build and run workflow
    graph = build_workflow()
    final_state = graph.invoke(initial_state)   
    return final_state

""" 
def _display_results(state: HouseholdProfileState):
  
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
 """

def main():   
    profile_1 = {
        'house_type': 'Apartment',
        'num_bedrooms': 2,
        'climate_zone': 'Hot & Dry',
        'city_tier': 'Tier 3',
        'num_occupants': 3,
    }
    state_1 = initializeProfile(profile_1)   
    print("updated state", state_1)
    

if __name__ == "__main__":
    main()