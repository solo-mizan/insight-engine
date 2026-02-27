from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from .state import AgentState
from .tools import ingest_pdfs
from dotenv import load_dotenv
from .model_factory import get_model

load_dotenv()

# 1. initialize the LLM
llm = get_model()

# 2. define the node (the "work" functions)
def retrive_node(state: AgentState):
    print("---RETRIVING CONTEXT---")
    retriver = ingest_pdfs("./data")
    docs = retriver.invoke(state["question"])
    context = [doc.page_content for doc in docs]
    count = state.get("loop_count", 0) + 1
    return {"context": context, "steps": state["steps"] + ["retrived_context"]}

def generate_answer_node(state:AgentState):
    print("---GENERATING ANSWER (WITH FALLBACK PROTECTION)---")
    prompt = f"""
    Use the following context to answer the user's question.
    If you don't know, say you don't know based on the docs.
    Context: {state['context']}
    Question: {state['question']}
    """
    response = llm.invoke(prompt)
    return {"answer": response.content, "steps": state["steps"] + ["generated_answer"]}

def quality_check_node(state: AgentState):
    """
    The 'Critic' Node.
    We ask the LLM to verify if the answer is grounded in context.
    """
    print("---PERFORMING QUALITY CHECK---")
    if not state["context"]:
        return {"answer": "I couldn't find any relevant information in the documents", "steps": state["steps"] + ["failed_no_context"]}

    check_prompt = f"""
    Assess if the following Answer is grounded in the Context.
    Context: {state['context']}
    Answer: {state['answer']}
    
    Respond only with 'YES' if the answer is helpful and based on context, or 'NO' if it is a hallucination.
    """

    response = llm.invoke(check_prompt)
    is_valid = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper() # type: ignore

    if "YES" in is_valid:
        return {"steps": state["steps"] + ["passed_quality_check"]}
    else:
        return {"answer": "REDO", "steps": state["steps"] + ["failed_quality_check"]}
    
# 3. define the router logic
def decide_to_finish(state: AgentState):
    if state.get("loop_count", 0) >= 3:
        print("---MAX LOOPS REACHED: FORCING EXIT---")
        return END
    
    if state["answer"] == "REDO":
        return "retrieve" # loop back
    return END

# 4. build the graph
builder = StateGraph(AgentState)

builder.add_node("retrieve", retrive_node)
builder.add_node("generate", generate_answer_node)
builder.add_node("quality_check", quality_check_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "quality_check")

# conditional edge: this is the "decision" part of the brain
builder.add_conditional_edges(
    "quality_check",
    decide_to_finish
)

# compile the graph
research_app = builder.compile()