# ============================================================
# CrewAI Trading Platform
# ============================================================
#
# Required packages:
# pip install crewai
# pip install crewai-tools
# pip install langchain-openai
# pip install python-dotenv
#
# ============================================================

import os

from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

# Load .env file from the main project folder
ENV_FILE = r"C:\BUILDING GENERATIVE AI APPLICATIONS\.env"

load_dotenv(dotenv_path=ENV_FILE)


# ============================================================
# 2. READ API KEYS
# ============================================================

openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")


# ============================================================
# 3. CHECK API KEYS
# ============================================================

if not openai_api_key:
    raise ValueError(
        "\nOPENAI_API_KEY is missing.\n"
        "Please add OPENAI_API_KEY to:\n"
        f"{ENV_FILE}\n"
    )

if not serper_api_key:
    raise ValueError(
        "\nSERPER_API_KEY is missing.\n"
        "Please add SERPER_API_KEY to:\n"
        f"{ENV_FILE}\n"
    )


print("\nAPI keys loaded successfully.")


# ============================================================
# 4. CREATE TOOLS
# ============================================================

search_tool = SerperDevTool()

scrape_tool = ScrapeWebsiteTool()


# ============================================================
# 5. CREATE DATA ANALYST AGENT
# ============================================================

data_analyst_agent = Agent(
    role="Data Analyst",

    goal=(
        "Monitor and analyze market data for the selected stock "
        "to identify trends, important market information, "
        "and potential market movements."
    ),

    backstory=(
        "You are a financial data analyst specializing in "
        "financial markets, statistical analysis, and machine "
        "learning. You analyze available market information "
        "and provide structured insights that can be used "
        "by the other agents."
    ),

    verbose=True,

    allow_delegation=True,

    tools=[
        scrape_tool,
        search_tool
    ]
)


# ============================================================
# 6. CREATE TRADING STRATEGY AGENT
# ============================================================

trading_strategy_agent = Agent(
    role="Trading Strategy Developer",

    goal=(
        "Develop and evaluate trading strategies based on "
        "market insights provided by the Data Analyst."
    ),

    backstory=(
        "You have knowledge of financial markets and "
        "quantitative analysis. You examine market trends "
        "and develop hypothetical trading strategies while "
        "considering the specified risk tolerance and "
        "trading preferences."
    ),

    verbose=True,

    allow_delegation=True,

    tools=[
        scrape_tool,
        search_tool
    ]
)


# ============================================================
# 7. CREATE TRADE ADVISOR AGENT
# ============================================================

execution_agent = Agent(
    role="Trade Advisor",

    goal=(
        "Analyze potential trading strategies and provide "
        "hypothetical execution considerations based on "
        "market conditions."
    ),

    backstory=(
        "You specialize in evaluating trade timing, pricing, "
        "market conditions, and execution considerations. "
        "You provide analytical suggestions rather than "
        "executing actual trades."
    ),

    verbose=True,

    allow_delegation=True,

    tools=[
        scrape_tool,
        search_tool
    ]
)


# ============================================================
# 8. CREATE RISK ADVISOR AGENT
# ============================================================

risk_management_agent = Agent(
    role="Risk Advisor",

    goal=(
        "Evaluate the risks associated with potential "
        "trading strategies and provide risk-mitigation insights."
    ),

    backstory=(
        "You specialize in financial risk assessment and "
        "market risk analysis. You examine potential risks, "
        "uncertainties, volatility, and downside scenarios "
        "associated with hypothetical trading strategies."
    ),

    verbose=True,

    allow_delegation=True,

    tools=[
        scrape_tool,
        search_tool
    ]
)


# ============================================================
# 9. CREATE DATA ANALYSIS TASK
# ============================================================

data_analysis_task = Task(
    description=(
        "Analyze available market data for the selected stock "
        "({stock_selection}). Use statistical reasoning and "
        "available market information to identify relevant "
        "trends, price movements, news events, and potential "
        "market opportunities or threats."
    ),

    expected_output=(
        "A structured market analysis report for {stock_selection} "
        "including current market information, important trends, "
        "relevant news, and potential opportunities or risks."
    ),

    agent=data_analyst_agent
)


# ============================================================
# 10. CREATE STRATEGY DEVELOPMENT TASK
# ============================================================

strategy_development_task = Task(
    description=(
        "Develop hypothetical trading strategies for "
        "{stock_selection} based on the Data Analyst's findings. "
        "Consider the user's risk tolerance ({risk_tolerance}) "
        "and trading preference ({trading_strategy_preference}). "
        "Clearly explain the assumptions behind each strategy."
    ),

    expected_output=(
        "A structured set of hypothetical trading strategies "
        "for {stock_selection}, including strategy logic, "
        "entry considerations, exit considerations, assumptions, "
        "and major risks."
    ),

    agent=trading_strategy_agent,

    context=[
        data_analysis_task
    ]
)


# ============================================================
# 11. CREATE EXECUTION PLANNING TASK
# ============================================================

execution_planning_task = Task(
    description=(
        "Analyze the hypothetical trading strategies developed "
        "for {stock_selection}. Provide execution considerations "
        "based on market conditions, liquidity, timing, pricing, "
        "and the user's stated trading preference."
    ),

    expected_output=(
        "A detailed hypothetical execution plan for "
        "{stock_selection}, including timing considerations, "
        "pricing considerations, market conditions, and "
        "potential execution risks."
    ),

    agent=execution_agent,

    context=[
        data_analysis_task,
        strategy_development_task
    ]
)


# ============================================================
# 12. CREATE RISK ASSESSMENT TASK
# ============================================================

risk_assessment_task = Task(
    description=(
        "Evaluate the risks associated with the proposed "
        "hypothetical trading strategies and execution plans "
        "for {stock_selection}. Consider market volatility, "
        "liquidity, news risk, downside scenarios, and the "
        "specified risk tolerance ({risk_tolerance})."
    ),

    expected_output=(
        "A comprehensive risk analysis report for "
        "{stock_selection}, including major risks, potential "
        "downside scenarios, risk indicators, and possible "
        "risk-mitigation approaches."
    ),

    agent=risk_management_agent,

    context=[
        data_analysis_task,
        strategy_development_task,
        execution_planning_task
    ]
)



# ============================================================
# 14. CREATE CREW
# ============================================================

financial_trading_crew = Crew(

    agents=[
        data_analyst_agent,
        trading_strategy_agent,
        execution_agent,
        risk_management_agent
    ],

    tasks=[
        data_analysis_task,
        strategy_development_task,
        execution_planning_task,
        risk_assessment_task
    ],

    manager_llm="gpt-4o-mini",

    process=Process.hierarchical,

    verbose=True
)


# ============================================================
# 15. INPUT DATA
# ============================================================

financial_trading_inputs = {

    "stock_selection": "AAPL",

    "initial_capital": "100000",

    "risk_tolerance": "Medium",

    "trading_strategy_preference": "Day Trading",

    "news_impact_consideration": True
}


# ============================================================
# 16. RUN CREW
# ============================================================

print("\n")
print("=" * 70)
print("STARTING CREWAI FINANCIAL TRADING ANALYSIS")
print("=" * 70)
print("\n")

result = financial_trading_crew.kickoff(
    inputs=financial_trading_inputs
)


# ============================================================
# 17. DISPLAY FINAL RESULT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)
print("\n")

print(result)


# ============================================================
# 18. SAVE RESULT TO MARKDOWN FILE
# ============================================================

output_file = r"C:\Users\malah\Downloads\output.md"

with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(result.raw)


print("\n")
print("=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print(f"\nMarkdown output saved to:")
print(output_file)