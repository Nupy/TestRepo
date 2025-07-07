from crewai import Crew, Agent, Task

# Define the agents responsible for each part of the analysis
lister_agent = Agent(
    role="Financial Analyst",
    goal="Determine if a given company is publicly listed",
    backstory="You are an expert at analyzing stock listings."
)

research_agent = Agent(
    role="Market Researcher",
    goal="Collect information about the company including financial outlook",
    backstory="You gather data from trustworthy financial sources."
)

invest_agent = Agent(
    role="Investment Advisor",
    goal="Provide an investment recommendation based on the research",
    backstory="You analyze the company's outlook and decide if it is a good investment."
)

# Tasks for each agent
list_task = Task(
    description="Check if {company} is publicly listed and provide the ticker if available.",
    expected_output="A short note about the listing status.",
    agent=lister_agent
)

research_task = Task(
    description="Research {company}'s financials, market position, and outlook.",
    expected_output="A brief summary of the company.",
    agent=research_agent
)

recommend_task = Task(
    description=(
        "Given {company}, its listing status, and the research summary, provide a recommendation "
        "on whether to invest in the company."
    ),
    expected_output="A clear yes/no investment recommendation with reasoning.",
    agent=invest_agent
)

# Assemble the crew with all agents and tasks
crew = Crew(
    agents=[lister_agent, research_agent, invest_agent],
    tasks=[list_task, research_task, recommend_task]
)

def analyze_company(company_name: str) -> str:
    """Run the Crew AI workflow for a given company."""
    results = crew.kickoff(inputs={"company": company_name})
    return results

if __name__ == "__main__":
    company = input("Enter company name: ")
    print(analyze_company(company))
