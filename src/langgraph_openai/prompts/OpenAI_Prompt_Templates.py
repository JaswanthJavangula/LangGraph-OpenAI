from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate



simpleChatPrompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful chat assistant."),
    ("user", "{question}")
])

def getResearchPydanticPrompt(parser: PydanticOutputParser) -> ChatPromptTemplate:

    return ChatPromptTemplate.from_messages([
        ("system", """You are a helpful research assistant.
                   Answer the question with simple research and tools.
                   Wrap the output in this format and provide no additional text\n{pydantic_parsing_format}
                   """),
        ("human", "{question}")
    ]).partial(pydantic_parsing_format=parser.get_format_instructions())

