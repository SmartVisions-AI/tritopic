"""
LLM-based Topic Labeling
=========================

Generate human-readable topic labels using LLMs:
- Claude (Anthropic)
- GPT-4 (OpenAI)
"""

from __future__ import annotations

from typing import Literal
import json


class LLMLabeler:
    """
    Generate topic labels using Large Language Models.
    
    Uses LLMs to create meaningful, human-readable labels for topics
    based on their keywords and representative documents.
    
    Parameters
    ----------
    provider : str
        LLM provider: "anthropic" or "openai"
    api_key : str
        API key for the provider.
    model : str, optional
        Model name. Defaults to best available model.
    max_tokens : int
        Maximum tokens in response. Default: 200
    temperature : float
        Sampling temperature. Default: 0.3
    language : str
        Output language. Default: "english"
    """
    
    def __init__(
        self,
        provider: Literal["anthropic", "openai"] = "anthropic",
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.3,
        language: str = "english",
        domain_hint: str | None = None,
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model or self._default_model()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.language = language
        self.domain_hint = domain_hint

        self._client = None
    
    def _default_model(self) -> str:
        """Get default model for provider."""
        if self.provider == "anthropic":
            return "claude-haiku-4-5-20251001"
        else:
            return "gpt-4o-mini"
    
    def _init_client(self):
        """Initialize API client."""
        if self._client is not None:
            return
        
        if self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
        else:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Install with: pip install openai"
                )
    
    def generate_label(
        self,
        keywords: list[str],
        representative_docs: list[str],
        domain_hint: str | None = None,
    ) -> tuple[str, str]:
        """
        Generate a label for a topic.

        Parameters
        ----------
        keywords : list[str]
            Topic keywords (top 10 recommended).
        representative_docs : list[str]
            Representative documents for the topic.
        domain_hint : str, optional
            Domain context (e.g., "tourism", "technology").

        Returns
        -------
        label : str
            Short topic label (2-5 words).
        description : str
            Brief description of the topic.
        """
        self._init_client()

        # Build prompt
        system_prompt, user_prompt = self._build_prompt(
            keywords, representative_docs, domain_hint
        )

        # Call API with error handling
        try:
            if self.provider == "anthropic":
                response = self._call_anthropic(system_prompt, user_prompt)
            else:
                response = self._call_openai(system_prompt, user_prompt)
        except Exception as e:
            # Fallback to keyword-based label on API error
            import warnings
            warnings.warn(f"LLM API call failed: {e}. Falling back to keyword label.")
            fallback_label = " & ".join(kw.title() for kw in keywords[:3])
            return fallback_label, f"Topic related to: {', '.join(keywords[:6])}"

        # Parse response
        label, description = self._parse_response(response)

        return label, description
    
    def _build_prompt(
        self,
        keywords: list[str],
        representative_docs: list[str],
        domain_hint: str | None = None,
    ) -> tuple[str, str]:
        """Build the labeling prompt.

        Returns
        -------
        system_prompt : str
            System-level instruction for the LLM.
        user_prompt : str
            User-level prompt with topic data.
        """
        # Truncate long documents
        docs_text = ""
        for i, doc in enumerate(representative_docs[:5], 1):
            truncated = doc[:500] + "..." if len(doc) > 500 else doc
            docs_text += f"\nDocument {i}: {truncated}\n"

        hint = domain_hint or self.domain_hint
        domain_context = ""
        if hint:
            domain_context = f"\nDomain context: This is about {hint}.\n"

        system_prompt = (
            "You are an expert at creating concise, meaningful topic labels for topic modeling. "
            "You always respond with valid JSON and nothing else."
        )

        user_prompt = f"""Given the following information about a topic, create:
1. A SHORT LABEL (2-5 words, title case, no special characters)
2. A BRIEF DESCRIPTION (1-2 sentences explaining what this topic is about)

Keywords (most representative words for this topic):
{', '.join(keywords[:10])}

Representative Documents:
{docs_text}
{domain_context}
Requirements:
- The label should be specific and descriptive, not generic
- The label should capture the main theme, not just list keywords
- The description should explain what documents in this topic discuss
- Output in {self.language}

Respond ONLY with this exact JSON format, no other text:
{{"label": "Your Topic Label", "description": "Your brief description."}}"""

        return system_prompt, user_prompt
    
    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Call Anthropic API."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API."""
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
    
    def _parse_response(self, response: str) -> tuple[str, str]:
        """Parse LLM response to extract label and description."""
        import re

        if not response or not response.strip():
            return "Unknown Topic", ""

        # Strategy 1: Try to parse complete JSON from response
        try:
            start = response.find("{")
            end = response.rfind("}") + 1

            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                label = data.get("label", "Unknown Topic")
                description = data.get("description", "")

                return str(label).strip(), str(description).strip()
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Strategy 2: Handle truncated JSON by extracting fields with regex
        label_match = re.search(r'"label"\s*:\s*"([^"]*)"', response)
        desc_match = re.search(r'"description"\s*:\s*"([^"]*)', response)

        if label_match:
            label = label_match.group(1).strip()
            description = desc_match.group(1).strip() if desc_match else ""
            return label, description

        # Strategy 3: Fallback - extract from plain text
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        label = lines[0] if lines else "Unknown Topic"
        description = " ".join(lines[1:]) if len(lines) > 1 else ""

        # Clean up common artifacts
        for prefix in ["Label:", "label:", "**Label:**", "1.", "1)"]:
            if label.startswith(prefix):
                label = label[len(prefix):].strip()
        for prefix in ["Description:", "description:", "**Description:**", "2.", "2)"]:
            if description.startswith(prefix):
                description = description[len(prefix):].strip()

        label = label.strip('"\'{}')
        description = description.strip('"\'{}')

        return label or "Unknown Topic", description
    
    def generate_labels_batch(
        self,
        topics_data: list[dict],
        domain_hint: str | None = None,
    ) -> list[tuple[str, str]]:
        """
        Generate labels for multiple topics.
        
        Parameters
        ----------
        topics_data : list[dict]
            List of dicts with "keywords" and "representative_docs".
        domain_hint : str, optional
            Domain context.
            
        Returns
        -------
        labels : list[tuple[str, str]]
            List of (label, description) tuples.
        """
        results = []
        
        for topic in topics_data:
            label, desc = self.generate_label(
                keywords=topic["keywords"],
                representative_docs=topic["representative_docs"],
                domain_hint=domain_hint,
            )
            results.append((label, desc))
        
        return results


class SimpleLabeler:
    """
    Simple rule-based labeler (no LLM required).
    
    Creates labels from top keywords.
    """
    
    def __init__(self, n_words: int = 3):
        self.n_words = n_words
    
    def generate_label(
        self,
        keywords: list[str],
        **kwargs,
    ) -> tuple[str, str]:
        """Generate label from top keywords."""
        # Take top n keywords
        top_keywords = keywords[:self.n_words]
        
        # Title case
        label = " & ".join(kw.title() for kw in top_keywords)
        
        # Description from more keywords
        description = f"Topics related to: {', '.join(keywords[:6])}"
        
        return label, description
