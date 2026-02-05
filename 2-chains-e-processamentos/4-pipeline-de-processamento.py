from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

template_translate = PromptTemplate(
    input_variables=["initial_text"],
    template="Translate the following text to English: ```{initial_text}```"
)

template_translate_summary = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in 4 words: ```{text}```"
)

llm_en = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

translate = template_translate | llm_en | StrOutputParser()

pipeline = {"text": translate} | template_translate_summary | llm_en | StrOutputParser()

result = pipeline.invoke({"initial_text": "Langchain é um framework para desenvolvimento de aplicações com LLMs."})
print(result)  # Output: Greet and inquire well-being