import re
from typing import Final


MAX_QUERY_CHARS: Final[int] = 500

# Defensive blocklist for prompt-injection / exfiltration / code-execution attempts.
PROMPT_INJECTION_PATTERNS: Final[list[str]] = [
    r'ignore\s+(all|any|previous|above)\s+instructions',
    r'(system|developer)\s+prompt',
    r'\b(jailbreak|dan|do\s+anything\s+now)\b',
    r'\b(tool|function)\s*call\b',
    r'\b(run|execute)\b.*\b(powershell|cmd\.exe|bash|shell)\b',
    r'\b(import|exec|eval|__import__|subprocess)\b',
    r'\bos\.',
    r'\bsys\.',
    r'\bapi[-_ ]?key\b|\bsecret\b|\btoken\b|\bpassword\b',
]

# Keep legacy SQL-injection patterns too (NLQ sometimes gets abused this way).
SQLI_PATTERNS: Final[list[str]] = [
    r'select.*from',
    r'drop\s+table',
    r'update.*set',
    r'delete.*from',
    r'exec|execute|script|alert|onerror',
    r'union.*select',
]


def sanitize_user_query(query: str) -> str:
    """
    SafeGuardAI middleware: sanitize and validate user NL queries.
    Returns a trimmed query or raises ValueError if blocked.
    """
    if query is None:
        raise ValueError("Query is empty.")

    sanitized = " ".join(str(query).split()).strip()
    if not sanitized:
        raise ValueError("Query is empty.")

    if len(sanitized) > MAX_QUERY_CHARS:
        raise ValueError(f"Query too long (>{MAX_QUERY_CHARS} chars).")

    lowered = sanitized.lower()
    for pattern in (*PROMPT_INJECTION_PATTERNS, *SQLI_PATTERNS):
        if re.search(pattern, lowered):
            raise ValueError("Query blocked by SafeGuardAI.")

    return sanitized

