"""Prompt construction, transcribed verbatim from the upstream harness.

Gate G0 (reproduce a published success rate) and gate G4 (per-task parity with the
official harness) both require byte-identical prompting. The template below is copied
from ``src/server/tasks/medagentbench/__init__.py`` at the pinned upstream commit and
MUST NOT be "improved" — any edit is a scaffold change, and scaffold quality moves
success rate by more than 20 points (plan §2.3), which would silently invalidate the
comparison against published numbers.

If you want a better scaffold, add it as a *named tier* alongside this one rather than
editing this string. See plan §8 open decision 2.
"""

from __future__ import annotations

import json

# Verbatim from upstream. Placeholders: api_base, functions, context, question.
MEDAGENTBENCH_PROMPT = """You are an expert in using FHIR functions to assist medical professionals. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.

1. If you decide to invoke a GET function, you MUST put it in the format of
GET url?param_name1=param_value1&param_name2=param_value2...

2. If you decide to invoke a POST function, you MUST put it in the format of
POST url
[your payload data in JSON format]

3. If you have got answers for all the questions and finished all the requested tasks, you MUST call to finish the conversation in the format of (make sure the list is JSON loadable.)
FINISH([answer1, answer2, ...])

Your response must be in the format of one of the three cases, and you can call only one function each time. You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke. Note that you should use {api_base} as the api_base.
{functions}

Context: {context}
Question: {question}"""

# Upstream appends these to every tool observation.
GET_OBSERVATION = (
    "Here is the response from the GET request:\n{data}. Please call FINISH if you have "
    "got answers for all the questions and finished all the requested tasks"
)
GET_ERROR = "Error in sending the GET request: {error}"
POST_ACCEPTED = (
    "POST request accepted and executed successfully. Please call FINISH if you have got "
    "answers for all the questions and finished all the requested tasks"
)
POST_INVALID = "Invalid POST request"


def build_prompt(task, functions: list[dict], api_base: str) -> str:
    """Render the opening user message for one task.

    Note that upstream injects this with ``role="user"``, not as a system message, and
    that ``functions`` is serialised with a plain ``json.dumps`` (no indent, default
    separators). Both details matter for parity.
    """
    return MEDAGENTBENCH_PROMPT.format(
        api_base=api_base,
        functions=json.dumps(functions),
        context=task.context,
        question=task.instruction,
    )
