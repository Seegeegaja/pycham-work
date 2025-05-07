import json

from openai import Client
import config
from config import OPEN_API_KEY
from inference import client
from prompt_template import prompt_function_calling

client = Client(api_key=OPEN_API_KEY)


def predict_function_calling(review):
    prompt = prompt_function_calling.format(review=review)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "positive_and_negative_keywords",
                "description": "Extract_keyword",
                "parameters":
                    {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "properties": {
                            "positive_keywords": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "negative_keywords": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "required": ["positive_keywords", "negative_keywords"]
                    }


            }
        }

    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system" , "content" : "You are a helpful assistant"},
            {"role":"user" , "content":prompt}
        ],
        temperature=0,
        tools=tools,
        tool_choice={
            "type":"function",
            "function":{
                "name":"positive_and_negative_keywords"
            }
        }
    )

    output = response.choices[0].message.tool_calls[0].function.arguments
    output_json = json.loads(output)
    return output_json
if __name__ == '__main__':
    review_1 = "보는 내내 시간가는줄 모르고 정말 재밋게 봤어요"
    review_2 = "정말 쓰레기 같은 영화 절대비추 음악은 그나마 좋은편"
    print(predict_function_calling(review_1))
    print(predict_function_calling(review_2))
