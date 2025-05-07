from http.client import responses

from numpy.ma.core import outer
from openai import Client

import config
import prompt_template
from inference import client

client = Client(api_key=config.OPEN_API_KEY)

def inference_json(review):
    prompt = prompt_template.prompt_question_json.format(review=review)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system", "content":"You are a helpful assistant"},
            {"role":"user" , "content":prompt}
        ],
        temperature=0,
        response_format={"type":"json_object"}
    )
    output = response.choices[0].message.content
    return output
def predict_keyword(review):
    prompt = prompt_template.prompt_keyword.format(review=review)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system" , "content":"You are a helpful assistant"},
            {"role":"user", "content":prompt}
        ],
        temperature=0

    )
    output = response.choices[0].message.content
    print(output)


if __name__ == '__main__':
    review = "보는 내내 시간 가는줄 모르고 정말 재밌게 봤습니다"
    predict_keyword(review)
    # print(inference_json(review))

