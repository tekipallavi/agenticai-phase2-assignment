from typing import TypedDict, Optional, List, Dict, Any, Literal


class HouseholdProfileState(TypedDict, total=False):
    profile_data: Dict
    messages: List[str]
    errors: List[str]
    current_agent: Optional[str]
    workflow_stage: Literal['Start', 'In progress', 'Error', 'Complete']
