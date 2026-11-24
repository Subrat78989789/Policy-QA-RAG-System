
from llm import LLM

llm = LLM()

answer = llm.generate_answer(
    question="When does illness coverage begin?",
    context="""
    The policy has a 30-day initial waiting
    period for illnesses.
    Accidental injuries are covered from
    the policy start date.
    """
)

print("\nLLM Answer:")
print(answer)
