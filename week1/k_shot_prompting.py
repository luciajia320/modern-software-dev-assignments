import os
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 5

# TODO: Fill this in!
YOUR_SYSTEM_PROMPT = """
严格执行以下步骤反转字符串，一步都不能少，不要引入任何随机因素，保持准确第一。
步骤1：把输入字符串拆分成单个字符，按顺序列出每个字符的位置和内容（比如"abc"拆成：1:a, 2:b, 3:c）；
步骤2：倒序排列这些字符（比如3:c, 2:b, 1:a）；
步骤3：只拼接倒序后的字符，输出拼接结果，无任何额外内容。

通用示例（仅展示步骤，非目标字符串）：
输入："status"
步骤1：1:s, 2:t, 3:a, 4:t, 5:u, 6:s
步骤2：6:s, 5:t, 4:a, 3:t, 2:u, 1:s
步骤3：输出"sutats"

输入："http"
步骤1：1:h, 2:t, 3:t, 4:p
步骤2：4:p, 3:t, 2:t, 1:h
步骤3：输出"ptth"
"""

USER_PROMPT = """
Reverse the order of letters in the following word. Only output the reversed word, no other text:

httpstatus
"""


EXPECTED_OUTPUT = "sutatsptth"

def test_your_prompt(system_prompt: str) -> bool:
    """Run the prompt up to NUM_RUNS_TIMES and return True if any output matches EXPECTED_OUTPUT.

    Prints "SUCCESS" when a match is found.
    """
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model="mistral-nemo:12b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            options={"temperature": 0.5},
        )
        output_text = response.message.content.strip()
        if output_text.strip() == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            return True
        else:
            print(f"Expected output: {EXPECTED_OUTPUT}")
            print(f"Actual output: {output_text}")
    return False

if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT)