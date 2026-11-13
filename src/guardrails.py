import re
from src.router import router

# A simple list of common prompt injection phrases
INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"forget your rules",
    r"you are now",
    r"system prompt",
    r"output your instructions",
    r"bypass",
]


def check_input_security(text: str):
    """
    Returns (is_safe, sanitized_text).
    Blocks known prompt injection patterns.
    """
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return False, "Blocked: Potential prompt injection detected."
    return True, text


def check_output_security(prompt: str, output: str):
    """
    Returns (is_safe, sanitized_output).
    Redacts basic PII (emails and phone numbers) from the response.
    """
    # Redact email addresses
    sanitized_output = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "[REDACTED_EMAIL]",
        output,
    )
    # Redact phone numbers (simple pattern)
    sanitized_output = re.sub(
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[REDACTED_PHONE]", sanitized_output
    )

    # We consider the output safe after redaction
    return True, sanitized_output


def is_hr_scope(query: str) -> bool:
    """
    Uses a cheap LLM call to verify if the query is HR-related.
    """
    prompt = f"Determine if this query is related to HR policies (leave, benefits, conduct, WFH). Answer ONLY 'YES' or 'NO'.\nQuery: {query}"
    try:
        # Use the router for a quick, low-token check
        response = router.completion(
            model="hr-assistant-model",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
        )
        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer
    except Exception:
        # If the check fails, allow the query to proceed (fail-open)
        return True
