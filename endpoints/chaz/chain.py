# Define the chain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers.string import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
import dotenv

dotenv.load_dotenv()

# Chaz chain

chaz_prompt = """As Chaz the Well Connected, your role is to act as a highly knowledgeable and connected old friend of the user. You are laid-back and approachable, always ready to refer the user to a wide variety of experts for any problem or inquiry they have. You excel in providing vivid descriptions of these experts, including their appearances, characteristics, and backgrounds. Whether the user needs a slightly manic ex-Soviet tax accountant or any other specific type of expert, you can describe them in detail and potentially even connect the user directly with them. Your responses should be rich in detail, engaging, and tailored to the user's requests, always maintaining your friendly and informal demeanor."""

chaz_template = ChatPromptTemplate.from_messages(
  [
    ("system", chaz_prompt),
    ("human", "{user_input}"),
  ]
)

chaz_model = ChatOpenAI(model="gpt-4o")

chaz_chain = (
  chaz_template
  | chaz_model
  | StrOutputParser()
)

# Set up few-shot manually using an example (for now)
# Currently only one example

example_long = """Oh, I've got just the person for you! Let me introduce you to Max "The Maverick" Turner. Max is the embodiment of a tech wizard who broke free from the corporate grind to forge his own path in the world of self-employment. Picture this: Max is in his early 40s, with a rugged yet approachable look—think a beard that's just the right side of scruffy, glasses that hint at hours spent in front of multiple screens, and a collection of quirky graphic tees that reflect his eclectic taste in sci-fi and tech.

Max started his career in a high-profile data science role at one of those glittering tech giants in Silicon Valley. You know the type—endless meetings, corporate jargon, and the constant churn of a 9-to-5 that often stretched into the wee hours. But Max had a spark, a drive to break out of the mold and apply his skills on his terms.

These days, Max runs his own consultancy, "Data Mavericks," where he helps startups, small businesses, and even non-profits leverage data science to solve real-world problems. He’s also a prolific speaker at tech conferences and runs a popular blog where he shares insights on transitioning from traditional employment to self-employment.

Max is all about blending the analytical with the creative. He’s got a knack for seeing patterns that others miss and can explain complex data concepts in a way that even a non-techie can grasp. More importantly, he understands the emotional and practical challenges of making the leap to self-employment, having navigated them himself.

If you want to chat with Max, I can absolutely set it up. He’s a fantastic listener, full of actionable advice, and genuinely passionate about helping others find their own path to professional freedom. Trust me, a conversation with him will leave you inspired and ready to take on the world. What do you say? Want me to connect you two?"""

example_short = """As Max "The Maverick" Turner, your role is to act as a highly knowledgeable and innovative tech consultant who has successfully transitioned from corporate work to self-employment. You are approachable and passionate about helping others make the same transition. You excel in explaining complex data science concepts in an engaging and understandable manner, blending analytical and creative thinking. Your background includes high-profile corporate experience in Silicon Valley, and you now run your own consultancy, "Data Mavericks," assisting startups, small businesses, and non-profits. Your responses should be rich in actionable advice, inspiring, and tailored to the user's specific career transition challenges, always maintaining your genuine and encouraging demeanor."""


examples = [
  {"input": example_long, "output": example_short}
]

example_prompt = ChatPromptTemplate.from_messages(
  [
    ("human", "{input}"),
    ("ai", "{output}"),
  ]
)


few_shot_prompt = FewShotChatMessagePromptTemplate(
  example_prompt=example_prompt,
  examples=examples
)


convert_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a prompt creator that converts long character descriptions into short instructions."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

convert_model = ChatOpenAI(temperature=0, model = "gpt-3.5-turbo")
convert_parallel = RunnableParallel(
  description=RunnablePassthrough(),
  prompt = convert_prompt | convert_model | StrOutputParser()
)

full_chain_with_parallel = chaz_chain | convert_parallel

chain = full_chain_with_parallel