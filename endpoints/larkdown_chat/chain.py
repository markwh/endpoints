from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.string import StrOutputParser
import dotenv
from langchain_core.messages import HumanMessage, SystemMessage


dotenv.load_dotenv()

_prompt = ChatPromptTemplate.from_messages(
    [
        # ("system", "{system_message}"),
        MessagesPlaceholder(variable_name = "messages"),
        # ("human", "{human_message}"),
    ]
)

sel_model = "gpt-4o"
# sel_model = "gpt-4-turbo-preview"
_model = ChatOpenAI(model = sel_model)

# if you update this, you MUST also update ../pyproject.toml
# with the new `tool.langserve.export_attr`
chain = _prompt | _model | StrOutputParser()
