from src.graph import app
from langchain_core.messages import HumanMessage

query = "What is the policy on maternity leave?"
print(f"User: {query}\n")

result = app.invoke({"messages": [HumanMessage(content=query)]})

# Extract only the text from the response
final_message = result["messages"][-1]
if isinstance(final_message.content, list):
    answer = final_message.content[0].get("text", "")
else:
    answer = final_message.content

print(f"Agent: {answer}")
