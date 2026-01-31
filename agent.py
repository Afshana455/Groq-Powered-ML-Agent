from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tool import predict_tool
import os

llm = ChatGroq(groq_api_key= os.getenv("GROQ_API_KEY"), model_name="qwen/qwen3-32b")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI agent that predicts outputs using an ML model."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(
    llm=llm,
    tools=[predict_tool],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[predict_tool],
    verbose=True
)

response = agent_executor.invoke({
    "input": "Use ML_Predictor with features=[332.5, 142.5, 0, 228, 0, 932, 594, 270]"})

print(response["output"])
