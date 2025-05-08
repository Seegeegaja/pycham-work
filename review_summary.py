import json
from openai import Client
from prompt_template import prompt_summary ,prompt_summary_langchain
import os
OPEN_API_KEY = os.getenv("OPEN_API_KEY")

client = Client(api_key=OPEN_API_KEY)

def summary(reviews):
    reviews = "\n".join(reviews)
    prompt = prompt_summary.format(reviews=reviews)

    respons = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[
            {"role":"system" ,"content":"You are a helpful assistant"},
            {"role":"user" , "content":prompt}
        ],
        temperature=0,
        response_format={"type":"json_object"}
    )
    output = respons.choices[0].message.content
    output_json = json.loads(output)
    return output_json



if __name__ == '__main__':
    print(summary(
        [
            "정말 재미있네요.",
            "시간가는줄 모르고 정말 즐겁게 봤습니다.",
            "다음 편 너무 기대 되요",
            "스케일 장난 아니고 배우들 연기도 훌륭했습니다."
        ]
    ))