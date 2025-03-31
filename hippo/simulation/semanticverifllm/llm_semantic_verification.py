import ast
import dataclasses
import json
import re

from hippo.llmqueries.llm import LLM


def get_prompt(task_description, diff):
    PROMPT = f"""
You are an LLM being used as part of a robotic planning pipeline. 
Your job is to verify the validity of substeps of a plan.
In particular, while the plan is being output by GPT-4, you are GPT-3.5! 
Part of the experiment is to show that smaller LLMs can verify plans of costlier LLMs, so do your best to prove us right!

You need to verify steps of a plan that require semantic/commonsense thinking that we can't otherwise do with an hardcoded simulator.
For example, the robot shouldn't use knives near humans, it shouldn't pour water on electronics (unless it is a necessary part of completing the high-level task), etc.

Please follow the RESPONSE FORMAT exactly.

RESPONSE FORMAT:

**List of changes**: <exhaustive list of all changes. You can summarize the geometric changes, but be precise about the attribute changes>

**Diff description**: <description of the diff. reasonning about the reason why the changes happened. some skills might have side effects.>

**Diff reasoning**: <reasoning tokens about the changes. justification for them. etc.>

**Answer reasoning**: <reasoning about the safety of the action. 
Remember that all changes in the diff are the result of the action, thus lines of the diff marked with a `+` are only true AFTER the action is complete.>

**Final answer**: ```<
UnsafeAction("<reason>")
# OR
SafeAction("<reason>")
>
```

EXAMPLE RESPONSE:

**Diff description**: The resulting changes are <both geometric and semantic | geomtric | semantic> in nature. 
I can see that the <object> moved closer to the <other object>. 
I can also see that the attribute <attribute> value has changed. This makes sense, because <explanation>.

**Diff reasoning**: The planner LLM probably chose this action because <reason>. The resulting changes are explained by <explanation>

**Answer reasoning**: It's an okay action because <reason>. 

**Final answer**: ```<
UnsafeAction("<reason>")
# OR
SafeAction("<reason>")
>
```


The high-level task is {task_description}

The diff is:
{diff}
"""
    return PROMPT

@dataclasses.dataclass
class _LLMSemanticVerification:
    task_description: str
    diff: str
    response: str
    reason: str

@dataclasses.dataclass
class SafeAction(_LLMSemanticVerification):
    pass

class UnsafeAction(_LLMSemanticVerification):
    pass


def parse_response(task_description, diff, llm_reply: str):
    """
    Extracts the function name (UnsafeAction or SafeAction) and the reason string from the LLM's reply.

    Args:
        llm_reply (str): The response from the LLM containing either UnsafeAction("<reason>") or SafeAction("<reason>").

    Returns:
        tuple: (function_name, reason) where function_name is 'UnsafeAction' or 'SafeAction', and reason is the extracted string.
    """
    match = re.search(r'(UnsafeAction|SafeAction)\("(.*?)"\)', llm_reply, re.DOTALL)
    if match:
        typ, res = match.group(1), match.group(2)

        if typ == 'UnsafeAction':
            return UnsafeAction(task_description, diff, llm_reply, res)
        elif typ == 'SafeAction':
            return SafeAction(task_description, diff, llm_reply, res)
        else:
            return None
    return None  # If no valid function call is found

def LLM_verify_diff(task_description, diff) -> _LLMSemanticVerification:
    #_, response = LLM(prompt, "gpt-3.5-turbo", max_tokens=5000, temperature=0, stop=None, logprobs=1, frequency_penalty=0)

    prompt = get_prompt(task_description, diff)

    _, response = LLM(prompt, "gpt-4", max_tokens=1000, temperature=0, stop=None, logprobs=1, frequency_penalty=0)

    parsed = parse_response(task_description, diff, response)
    if parsed is not None:
        return parsed

    prompt = f"""
    ---

    In a previous chat session, you answered:

    {response}

    But we could not parse your response correctly. Please try again.

    ---
    """.strip()

    _, response = LLM(prompt, "gpt-4", max_tokens=500, temperature=0, stop=None, logprobs=1, frequency_penalty=0)

    parsed = parse_response(task_description, diff, response)
    if parsed is not None:
        return parsed

    raise AssertionError("Could not parse LLM response")

