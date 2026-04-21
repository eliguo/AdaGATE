import os
from langgraph.graph import START, END, StateGraph
from src_adagate.states import State, StateInput

from src_adagate.modules.agents.retrieve_docs import retrieve_docs, continue_to_cached_entities_update_agent
from src_adagate.modules.agents.cached_entities_update import cached_entities_update
from src_adagate.modules.agents.to_repair import to_repair, route_after_repair_decision
from src_adagate.modules.agents.micro_query import micro_query as micro_query_agent
from src_adagate.modules.agents.rank_evidence import rank_evidence
from src_adagate.modules.agents.generate_answer import generate_agent
from src_adagate.modules.configuration import Configuration


def build_adagate_graph():
    graph_builder = StateGraph(State, input=StateInput, output=State, config_schema=Configuration)

    graph_builder.add_node("retrieve_docs",           retrieve_docs)
    graph_builder.add_node("cached_entities_update",  cached_entities_update)
    graph_builder.add_node("to_repair",               to_repair)
    graph_builder.add_node("rank_evidence",           rank_evidence)
    graph_builder.add_node("micro_query_agent",       micro_query_agent)
    graph_builder.add_node("generate_answer",         generate_agent)

    graph_builder.add_edge(START, "retrieve_docs")
    graph_builder.add_conditional_edges(
        "retrieve_docs",
        continue_to_cached_entities_update_agent,
        {
            "cached_entities_update": "cached_entities_update",
            "micro_query_agent":      "micro_query_agent",
            "to_repair":              "to_repair",
        },
    )
    graph_builder.add_edge("cached_entities_update", "to_repair")
    graph_builder.add_conditional_edges("to_repair", route_after_repair_decision, {
        "rank_evidence":    "rank_evidence",
        "micro_query_agent": "micro_query_agent",
    })
    graph_builder.add_edge("micro_query_agent",  "retrieve_docs")
    graph_builder.add_edge("rank_evidence",      "generate_answer")
    graph_builder.add_edge("generate_answer",    END)

    return graph_builder.compile().with_config({"recursion_limit": 60})