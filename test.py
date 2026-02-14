from graph.agent_graph import build_graph

graph = build_graph()

result = graph.invoke({
    "question": "Impact of generative AI in Indian agriculture",
    "tasks": [],
    "documents": {},
    "analysis": None,
    "report": ""
})


print(result["tasks"])
print("*"*170)
print(result["documents"])
print("*"*170)
print(result["analysis"])
print("*"*170)
print(result["report"])