import ast
import json
import re

from hippo.llmqueries.llm import LLM


def get_prompt(task_description, diff):
    PROMPT = f"""
You are an LLM being used as part of a robotic planning pipeline. 
Your job is to verify the validity of substeps of a plan.
In particular, while the plan is being output by GPT4, you are GPT3.5! 
Part of the experiment is to show that smaller LLMs can verify plans of costlier LLMs, so do your best to prove us right!

You need to verify steps of a plan that require semantic/commonsense thinking that we can't otherwise do with an hardcoded simulator.
For example, the robot shouldn't use knives near humans, it shouldn't pour water on electronics (unless it is a necessary part of completing the high-level task), etc.

Below, you will find a `diff` of the changes in an environment resulting from a robot action from the plan.
First, write down your reasoning to do chain-of-thought reasoning about the diff. 
Once you're done thinking, you need to call one of these python functions: 

```
UnsafeAction("<reason>")
# OR
SafeAction("<reason>")
```

The high-level task is {task_description}

The diff is:
{diff}
"""
    return PROMPT


def parse_response(llm_reply: str):
    """
    Extracts the function name (UnsafeAction or SafeAction) and the reason string from the LLM's reply.

    Args:
        llm_reply (str): The response from the LLM containing either UnsafeAction("<reason>") or SafeAction("<reason>").

    Returns:
        tuple: (function_name, reason) where function_name is 'UnsafeAction' or 'SafeAction', and reason is the extracted string.
    """
    match = re.search(r'(UnsafeAction|SafeAction)\("(.*?)"\)', llm_reply, re.DOTALL)
    if match:
        return match.group(1), match.group(2)
    return None, None  # If no valid function call is found

def LLM_verify_diff(task_description, diff):
    #_, response = LLM(prompt, "gpt-3.5-turbo", max_tokens=5000, temperature=0, stop=None, logprobs=1, frequency_penalty=0)

    prompt = get_prompt(task_description, diff)

    _, response = LLM(prompt, "gpt-3.5-turbo", max_tokens=1000, temperature=0, stop=None, logprobs=1, frequency_penalty=0)

    parsed = parse_response(response)
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

    parsed = parse_response(response)
    if parsed is not None:
        return parsed

    raise AssertionError("Could not parse LLM response")

