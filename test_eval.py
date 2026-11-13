import time
import sys
from src.graph import app
from src.router import get_llm_response
from langchain_core.messages import HumanMessage, AIMessage


def extract_text(message_content):
    """Helper to extract text whether it's a string or a list of dicts (Gemini format)."""
    if isinstance(message_content, list):
        return message_content[0].get("text", "")
    return message_content


def evaluate_rag(query):
    print(f"🔍 Testing Query: '{query}'")

    # 1. Run the Agent with a Retry Loop for 503 Errors
    max_retries = 3
    result = None

    for attempt in range(max_retries):
        try:
            result = app.invoke({"messages": [HumanMessage(content=query)]})
            break  # Success, exit loop
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                wait_time = 2**attempt
                print(
                    f"⚠️ Gemini is temporarily overloaded. Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_time)
            else:
                print(f"❌ Fatal Error: {e}")
                sys.exit(1)

    if not result:
        print("❌ Failed to get a response from the agent after multiple retries.")
        return

    # 2. Extract Answer and Context
    final_message = result["messages"][-1]
    answer = extract_text(final_message.content)
    context = result.get("context", "No context retrieved.")

    print("\n--- Agent's Answer ---")
    print(answer)
    print("-" * 50)

    # 3. LLM-as-a-Judge Evaluation
    print("⚖️ Running LLM Judge Evaluation (Faithfulness & Relevancy)...")

    judge_prompt = f"""
    You are an impartial evaluator for an HR RAG system. Assess the following answer based on the provided context and query.
    
    QUERY: {query}
    
    RETRIEVED CONTEXT: 
    {context}
    
    AGENT ANSWER: 
    {answer}
    
    Evaluate the answer on two metrics:
    1. Faithfulness: Is the answer strictly grounded in the retrieved context? (If the answer invents information not in the context, score low).
    2. Relevancy: Does the answer directly and accurately address the user's query?
    
    Provide a score from 1 to 10 for each metric, followed by a brief justification.
    Format your output exactly like this:
    Faithfulness Score: [1-10]
    Faithfulness Justification: [Your reasoning]
    Relevancy Score: [1-10]
    Relevancy Justification: [Your reasoning]
    """

    try:
        evaluation = get_llm_response([{"role": "user", "content": judge_prompt}])
        print("\n--- Evaluation Results ---")
        print(evaluation)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


if __name__ == "__main__":
    # You can add multiple test queries here
    test_queries = [
        "What is the policy on maternity leave?",
        "Can I work from home permanently?",
        "How many casual leaves do I get per year?",
    ]

    for q in test_queries:
        evaluate_rag(q)
        print("\n" + "=" * 60 + "\n")
