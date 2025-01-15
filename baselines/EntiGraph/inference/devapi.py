import os
import dotenv
from openai import AsyncOpenAI

dotenv.load_dotenv()

async def gptqa(prompt: str,
          openai_model_name: str,
          system_message: str,
          json_format: bool = False,
          temp: float = 1.0):
    client = AsyncOpenAI(
        api_key=os.getenv("TEACHER_API_KEY"),
        base_url=os.getenv("TEACHER_BASE_URL")
    )
    openai_model_name = openai_model_name or os.getenv("TEACHER_MODEL")

    if json_format:
        completion = await client.chat.completions.create(
            model=openai_model_name,
            temperature=temp,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system",
                "content": system_message},
                {"role": "user",
                "content": prompt},
            ])
    else:
        completion = await client.chat.completions.create(
            model=openai_model_name,
            temperature=temp,
            messages=[
                {"role": "system",
                "content": system_message},
                {"role": "user",
                "content": prompt},
            ])
    return completion.choices[0].message.content
