from graph.agent_graph import build_graph

graph = build_graph()

result = graph.invoke({
    "question": "Impact of generative AI in Indian agriculture",
    "tasks": [],
    "rewritten_tasks": None,
    "documents": {},
    "web_documents": {},
    "analysis": None,
    "retry_count": 0,
    "report": ""
})


print(result["tasks"])
print("*"*170)
print(result["documents"])
print("*"*170)
print(result["analysis"])
print("*"*170)
print(result["report"])