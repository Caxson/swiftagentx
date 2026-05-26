"""
Prompt template manager — manages agent prompt templates with YAML/JSON loading.
"""

import json
import re
from pathlib import Path

import yaml


class PromptTemplate:
    """A single prompt template with variable substitution."""

    def __init__(self, name: str, template: str, variables: list[str] | None = None):
        self.name = name
        self.template = template
        self.variables = variables or []

    def render(self, **kwargs: str) -> str:
        return self.template.format(**kwargs)

    def validate_variables(self, **kwargs: str) -> tuple:
        provided = set(kwargs.keys())
        required = set(self.variables)
        if required.issubset(provided):
            return True, []
        return False, list(required - provided)


class PromptManager:
    """
    Prompt manager — manages all prompt templates used by the agent.
    """

    def __init__(self) -> None:
        self.templates: dict[str, PromptTemplate] = {}
        self._setup_default_prompts()

    def _setup_default_prompts(self) -> None:
        self.register(
            "system_role",
            "You are an intelligent assistant agent with the following capabilities:\n"
            "- Understand and analyze user input\n"
            "- Solve problems through tool calls\n"
            "- Provide clear and accurate answers\n\n"
            "Principles:\n"
            "1. Be honest and accurate\n"
            "2. Show clear reasoning\n"
            "3. Be concise\n"
            "4. Refuse harmful requests",
        )

        self.register(
            "react_thought",
            "Iteration: {iteration}\n"
            "User input: {user_input}\n\n"
            "Available information:\n{context}\n\n"
            "Think:\n"
            "1. What does the user really need?\n"
            "2. What tool should I call?\n"
            "3. What should be the next step?\n\n"
            "Output format:\n"
            "Thought: [your reasoning]\n"
            "Tool: [tool name to call]",
        )

        self.register(
            "react_action",
            "I decided to use the {tool_name} tool.\n\n"
            "Tool input:\n{tool_input}\n\n"
            "This will help us {action_purpose}",
        )

        self.register("cache_hit", "Found relevant cached information:\n\n{cached_answer}")

        self.register(
            "final_answer",
            "Based on my analysis and tool results, here is the answer:\n\n{answer}\n\n"
            "Key info:\n- Time: {execution_time}ms\n- Tool calls: {tool_calls}\n- Sources: {sources}",
        )

        self.register(
            "output_prompt",
            "You are a friendly, professional intelligent assistant. "
            "Generate natural, friendly, and concise replies based on user input and relevant information.",
        )

    def register(self, name: str, template: str, variables: list[str] | None = None) -> PromptTemplate:
        if variables is None:
            variables = re.findall(r"\{(\w+)\}", template)
        prompt_template = PromptTemplate(name, template, variables)
        self.templates[name] = prompt_template
        return prompt_template

    def get(self, name: str) -> PromptTemplate | None:
        return self.templates.get(name)

    def render(self, template_name: str, **kwargs: str) -> str:
        template = self.get(template_name)
        if template is None:
            raise ValueError(f"Prompt template '{template_name}' not found")
        is_valid, missing = template.validate_variables(**kwargs)
        if not is_valid:
            raise ValueError(f"Missing variables for prompt '{template_name}': {missing}")
        return template.render(**kwargs)

    def render_safe(self, name: str, default: str = "", **kwargs: str) -> str:
        try:
            return self.render(name, **kwargs)
        except Exception:
            return default

    def list_templates(self) -> list[str]:
        return list(self.templates.keys())

    def load_from_yaml(self, file_path: str) -> None:
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._load_from_dict(data)

    def load_from_json(self, file_path: str) -> None:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        self._load_from_dict(data)

    def load_from_directory(self, directory: str, pattern: str = "*.yml") -> None:
        dir_path = Path(directory)
        for file_path in dir_path.glob(pattern):
            if file_path.suffix in (".yml", ".yaml"):
                self.load_from_yaml(str(file_path))
            elif file_path.suffix == ".json":
                self.load_from_json(str(file_path))

    def _load_from_dict(self, data: dict | None) -> None:
        if not isinstance(data, dict):
            return
        for name, template in data.items():
            if isinstance(template, str):
                self.register(name, template)
            elif isinstance(template, dict):
                self.register(name, template.get("template", ""), template.get("variables"))

    def get_output_prompt(self) -> str:
        template = self.get("output_prompt")
        if template:
            return template.template
        return "You are a helpful assistant."

    def export(self) -> dict[str, str]:
        return {name: t.template for name, t in self.templates.items()}

    def __len__(self) -> int:
        return len(self.templates)

    def __repr__(self) -> str:
        return f"<PromptManager: {len(self)} templates>"
