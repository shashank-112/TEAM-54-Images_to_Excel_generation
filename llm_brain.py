from openai import OpenAI

class LLMBrain:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("No API key provided")

        self.client = OpenAI(api_key=api_key)

    def interpret(self, user_text):
        prompt = f"""
Convert the user instruction into ONE command from this list:
- RUN_AGENT
- SHOW_STATUS
- SHOW_GOAL
- HELP
- EXIT

User input:
"{user_text}"

Only respond with the command.
"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You control an AI agent."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip()

