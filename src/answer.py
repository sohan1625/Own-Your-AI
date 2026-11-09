"""Answer generation module.

Responsible for:
- Constructing prompts containing strictly the retrieved context chunks, each identified by a unique chunk ID.
- Invoking the local language model (Mistral 7B Instruct via Ollama) with strict instructions to answer only from provided context.
- Enforcing explicit "I don't know" behavior when retrieved scores fall below threshold or context lacks the answer.
- Ensuring generated answers explicitly reference chunk IDs for every factual claim.
"""

