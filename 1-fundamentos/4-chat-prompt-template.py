from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


sytem = ("system", "You are an assistant that anwsers questions in a {style} style.")
user = ("user", "{question}")

chat_prompt = ChatPromptTemplate.from_messages([sytem, user])

messages = chat_prompt.format_messages(style="funny", question="What is LangChain?")

for msg in messages:
    print(f"{msg.type}: {msg.content}")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
response = model.invoke(messages)
print(response.content)