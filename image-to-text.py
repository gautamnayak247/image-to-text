from dotenv import load_dotenv, find_dotenv
from transformers import pipeline #download hugging face model to local machine
# from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai  import ChatOpenAI
import os

load_dotenv(find_dotenv())

def generate_context(url:str) -> str:
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)
    if text:
        return text[0]["generated_text"]
    else:
        return "No context found"

def generate_story_from_context(context: str) -> str:
    template = """ 
    You are a story teller:
    You can generate a short story based on a sample narrative, the story should be no more than 20 words;

    CONTEXT: {context}
    STORY:
    """
    
    prompt = PromptTemplate(template = template, input_variables = ["context"])
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature =1, openai_api_key=os.getenv("OPENAI_API_KEY"))

    story_llm  = LLMChain(llm= llm, prompt = prompt, verbose=True)
    story = story_llm.run(context = context)
    return story.strip()

def main():
    context = generate_context(".\\image\\xmas.jpg")
    story = generate_story_from_context(context)
    print(story)
    return

if __name__ == "__main__":
    main()