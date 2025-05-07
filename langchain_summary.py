from langchain.chains.question_answering.map_rerank_prompt import output_parser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import Client, api_key
from pydantic import BaseModel

from prompt_template import prompt_summary_langchain
from config import OPEN_API_KEY

api_key = OPEN_API_KEY

# json 결과를 담을 클레스 생성

class Output(BaseModel):
    summary: str

# Output parser 만들기 : LLM 결과를 class 맵핑
output_parser = PydanticOutputParser(pydantic_object=Output)

# prompt template을 생성
prompt_maker = PromptTemplate(
    template=prompt_summary_langchain,
    input_variables=["reviews"],
    partial_variables={
        "format_instruction": output_parser.get_format_instructions()
    }
)

print(prompt_maker)

model = ChatOpenAI(
    temperature=0,
    api_key = api_key,
    model_name = "gpt-4o-mini"
)
# langchain 처리
chain = (prompt_maker | model | output_parser)

def lang_chain_summary(reviews):
    reviews = "\n".join(reviews)
    # prompt 출력용
    prompt = prompt_maker.invoke({"reviews": reviews})
    output = chain.invoke({"reviews": reviews})
    return output.summary
if __name__ == '__main__':
    print(lang_chain_summary(
        [
            "정말 재미있네요.",
            "시간가는줄 모르고 정말 즐겁게 봤습니다.",
            "다음 편 너무 기대 되요",
            "스케일 장난 아니고 배우들 연기도 훌륭했습니다."
        ]
    ))