from typing import List, TypedDict

class AgentState(TypedDict):
    """
    Representing the state of our research assistant.
    """

    question: str
    context: List[str]
    answer: str
    steps: List[str] # a log of what the AI did (for transparency)
    loop_count: int