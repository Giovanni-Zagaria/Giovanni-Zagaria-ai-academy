import time
import streamlit as st


def safe_gpt_call(func, *args, max_retries: int = 5, wait_seconds: int = 60, **kwargs):
    """Gestisce il rate limit di Azure riprovando la chiamata."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:  # pragma: no cover - user feedback
            if hasattr(e, "status_code") and e.status_code == 429:
                st.warning(
                    f"Rate limit Azure superato! Aspetto {wait_seconds} secondi... (Tentativo {attempt+1}/{max_retries})"
                )
                time.sleep(wait_seconds)
            else:
                raise e
    raise Exception("Superato il numero massimo di retry per il rate limit.")
