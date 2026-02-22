from graph.agent_graph import build_graph

graph = build_graph()

result = graph.invoke({
    "question": "What is the effect of social media on the younger generations?",
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