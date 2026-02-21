from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents import create_agent



long_text = """
Once upon a time, in a small village nestled between rolling hills and lush forests, there lived a young girl named Lily. 
Lily was known for her curiosity and adventurous spirit. 
One day, while exploring the woods, she stumbled upon an old, abandoned mansion. 
Intrigued by the mystery surrounding the mansion, Lily decided to investigate further.
As she entered the mansion, she was greeted by a musty smell and the sound of creaking floorboards. 
The mansion was filled with old furniture, dusty books, and cobwebs hanging from the ceiling. 
Lily's heart raced with excitement as she explored each room, uncovering hidden secrets and forgotten treasures. 
In one of the rooms, she found a dusty old diary that belonged to a woman named Eleanor. 
As Lily read through the diary, she learned about Eleanor's life and the tragic events that led to her isolation in the mansion. 
Eleanor had been a talented artist who had fallen in love with a man named James. 
However, their love story was cut short when James was tragically killed in a car accident. 
Eleanor was devastated and withdrew from the world, choosing to live in the mansion alone. 
Lily felt a deep connection to Eleanor's story and decided to honor her memory by restoring the mansion and turning it into a community art center. 
With the help of the villagers, Lily worked tirelessly to clean and renovate the mansion, transforming it into a vibrant space for artists to showcase their work and for the community to come together. 
The art center became a hub of creativity and inspiration, attracting artists from all over the region. 
Lily's dedication and passion for art brought new life to the village, and the mansion that was once a symbol of isolation became a beacon of hope and creativity for everyone in the community. 
Lily's story is a testament to the power of resilience and the ability to find beauty and purpose in even the darkest of circumstances. 
Through her determination and love for art, Lily was able to transform a place of sorrow into a place of joy and inspiration, leaving a lasting legacy for generations to come.
"""

# Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# Split the long text into smaller chunks
chunks = text_splitter.split_text(long_text)
# Create a summarization middleware
summarization_middleware = SummarizationMiddleware(ChatGoogleGenerativeAI())
# Create an agent with the summarization middleware
agent = create_agent(middleware=[summarization_middleware])
# Use the agent to summarize each chunk of text
summarized_chunks = [agent.run(chunk) for chunk in chunks]
# Combine the summarized chunks into a single summary
final_summary = " ".join(summarized_chunks)
print(final_summary)

