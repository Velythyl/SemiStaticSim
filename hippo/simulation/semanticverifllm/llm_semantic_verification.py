import ast
import json

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


def parse_response(response):
    if "```" in response:
        splitted = response.split("```")[-2].strip()

        if splitted.count("[") != splitted.count("]"):
            return None


        if splitted.count("[") == 1:
            try:
                return json.loads(splitted)
            except:
                return ast.literal_eval(splitted)


        splitted = splitted.split("\n")
        ret = []
        for s in splitted:
            if "," in s:
                return None
            try:
                s = json.loads(s)
            except:
                s = ast.literal_eval(s)
            if len(s) > 1:
                return None
            ret.append(s[0])
        return ret

    return None

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

