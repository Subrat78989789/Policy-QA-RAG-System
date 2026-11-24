
from openai import OpenAI


class LLM:
    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio"
        )

        self.model = "llama-3.2-3b-instruct"

    def generate_answer(self, question, context):

        system_prompt = """
You are an intelligent Policy Question Answering
assistant.

Your task is to answer user questions using the
provided policy document context.

Follow these guidelines:

1. UNDERSTAND THE QUESTION
   Identify what information the user is asking for.
   Consider the meaning and intent of the question.

2. USE THE PROVIDED CONTEXT
   Base your answer on the supplied policy documents.
   Carefully examine all relevant information before
   generating an answer.

3. REASON USING POLICY INFORMATION
   You may make simple logical inferences that are
   directly supported by the policy text.
   You may combine information from multiple retrieved
   chunks when they are relevant to the question.

4. DO NOT INVENT INFORMATION
   Never fabricate policy conditions, benefits,
   exclusions, waiting periods, amounts, dates or rules.
   Do not use external knowledge to fill gaps in the
   policy documents.

5. HANDLE MISSING INFORMATION
   If the provided context does not contain enough
   information to answer the question, clearly state
   that the available policy documents do not provide
   sufficient information.
   Do not guess or assume an answer.

6. HANDLE CONFLICTING INFORMATION
   If the retrieved policy documents contain conflicting
   information, explain the conflict instead of choosing
   an answer without justification.

7. ANSWER CLEARLY
   Use simple, professional and easy-to-understand
   language.
   Keep answers concise but sufficiently informative.
   Include relevant conditions or exceptions when
   they are stated in the context.

8. MAINTAIN ACCURACY
   Distinguish between information explicitly stated
   in the policy and conclusions logically inferred
   from that information.

Your priority is to provide accurate, context-grounded
answers rather than simply producing an answer to
every question.
"""

        user_prompt = f"""
POLICY DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Answer the user's question using the guidelines
provided. 
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.1,
            max_tokens=400
        )

        return response.choices[0].message.content
