# Task for Nate: Building a Modern CrewAI Scheduling Assistant

Hey Nate, here's an updated guide to creating a personal scheduling assistant using CrewAI. This version uses a more modern, class-based approach that is better organized, more reusable, and aligns with the latest CrewAI practices.

## 1. Goal: A Modern, Reusable Scheduling Agent

We will build a self-contained `SchedulingAgent` class. This class will handle everything: initializing the AI model, defining the agent and its tools, and running the tasks. This is a much cleaner pattern than having loose functions.

**Core Concepts in this Approach:**

*   **Agent Class:** A Python class (`SchedulingAgent`) that encapsulates all the logic for our scheduling assistant.
*   **Class-Based Tool:** A robust way to create tools by inheriting from `BaseTool` and defining a `pydantic` model for the arguments. This provides structure, type safety, and clarity.
*   **`invoke` Method:** A single, clear entry point to make the agent perform its task.
*   **`crewai.LLM`:** The native CrewAI way to configure and use language models like Gemini.

## 2. Setting Up Your Environment

Let's get your project set up.

**Prerequisites:**

*   Python version 3.10 or newer.

**Installation:**

1.  **Create a project directory and a virtual environment:**

    ```bash
    mkdir nate-scheduling-agent
    cd nate-scheduling-agent
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install the necessary packages:**

    From the root of the project (`a2a-friend-scheduling`), you can now install Nate's agent and all its dependencies with a single command:
    ```bash
    pip install -e a2a_friend_scheduling/nate_agent
    ```
    This tells `pip` to install the `nate-scheduling-agent` package in "editable" (`-e`) mode, which is great for development.

3.  **Set up your Gemini API key:**

    Navigate to the `a2a_friend_scheduling/nate_agent` directory and create a file named `.env`.

    ```.env
    GOOGLE_API_KEY="your-google-api-key"
    ```

    You can get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## 3. Building the Scheduling Agent Class

Below is the complete, runnable Python script. Save this as `agent.py`. It implements the full `SchedulingAgent` using the modern, class-based approach with a well-defined tool.

```python
import os
import random
from datetime import date, datetime, timedelta
from typing import Type

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


def generate_calendar() -> dict[str, list[str]]:
    """Generates a random calendar for the next 7 days."""
    calendar = {}
    today = date.today()
    possible_times = [f"{h:02}:00" for h in range(8, 21)]  # 8 AM to 8 PM

    for i in range(7):
        current_date = today + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        available_slots = sorted(random.sample(possible_times, 8))
        calendar[date_str] = available_slots
    print("---- Nate's Generated Calendar ----")
    print(calendar)
    print("---------------------------------")
    return calendar


MY_CALENDAR = generate_calendar()


class AvailabilityToolInput(BaseModel):
    """Input schema for AvailabilityTool."""

    date_range: str = Field(
        ...,
        description="The date or date range to check for availability, e.g., '2024-07-28' or '2024-07-28 to 2024-07-30'.",
    )


class AvailabilityTool(BaseTool):
    name: str = "Calendar Availability Checker"
    description: str = (
        "Checks my availability for a given date or date range. "
        "Use this to find out when I am free."
    )
    args_schema: Type[BaseModel] = AvailabilityToolInput

    def _run(self, date_range: str) -> str:
        """Checks my availability for a given date range."""
        dates_to_check = [d.strip() for d in date_range.split("to")]
        start_date_str = dates_to_check[0]
        end_date_str = dates_to_check[-1]

        try:
            start = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            end = datetime.strptime(end_date_str, "%Y-%m-%d").date()

            if start > end:
                return "Invalid date range. The start date cannot be after the end date."

            results = []
            delta = end - start
            for i in range(delta.days + 1):
                day = start + timedelta(days=i)
                date_str = day.strftime("%Y-%m-%d")
                available_slots = MY_CALENDAR.get(date_str, [])
                if available_slots:
                    availability = f"On {date_str}, I am available at: {', '.join(available_slots)}."
                    results.append(availability)
                else:
                    results.append(f"I am not available on {date_str}.")

            return "\n".join(results)

        except ValueError:
            return (
                "I couldn't understand the date. "
                "Please ask to check availability for a date like 'YYYY-MM-DD'."
            )


class SchedulingAgent:
    """Agent that handles scheduling tasks."""

    def __init__(self):
        """Initializes the SchedulingAgent."""
        if os.getenv("GOOGLE_API_KEY"):
            self.llm = LLM(
                model="gemini/gemini-1.5-flash",
                api_key=os.getenv("GOOGLE_API_KEY"),
            )
        else:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")

        self.scheduling_assistant = Agent(
            role="Personal Scheduling Assistant",
            goal="Check my calendar and answer questions about my availability.",
            backstory=(
                "You are a highly efficient and polite assistant. Your only job is "
                "to manage my calendar. You are an expert at using the "
                "Calendar Availability Checker tool to find out when I am free. You never "
                "engage in conversations outside of scheduling."
            ),
            verbose=True,
            allow_delegation=False,
            tools=[AvailabilityTool()],
            llm=self.llm,
        )

    def invoke(self, question: str) -> str:
        """Kicks off the crew to answer a scheduling question."""
        task_description = (
            f"Answer the user's question about my availability. The user asked: '{question}'. "
            f"Today's date is {date.today().strftime('%Y-%m-%d')}."
        )

        check_availability_task = Task(
            description=task_description,
            expected_output="A polite and concise answer to the user's question about my availability, based on the calendar tool's output.",
            agent=self.scheduling_assistant,
        )

        crew = Crew(
            agents=[self.scheduling_assistant],
            tasks=[check_availability_task],
            process=Process.sequential,
            verbose=True,
        )
        result = crew.kickoff()
        return str(result)


if __name__ == "__main__":
    user_question = "Are you free for pickleball tomorrow?"
    scheduling_agent = SchedulingAgent()
    result = scheduling_agent.invoke(user_question)

    print("\n\n######################")
    print("## Here is the result")
    print("######################\n")
    print(result)
```

## 4. How to Run Your Agent

With your `agent.py` and `.env` files in place, run the script from your terminal:

```bash
python agent.py
```

This will instantiate your `SchedulingAgent` class and call its `invoke` method, kicking off the crew to answer the question.

## 5. Key Benefits of This Approach

*   **Encapsulation:** All the logic related to the agent is contained within a single class, making it easy to understand, manage, and import elsewhere in a larger application.
*   **Structured Tools:** Defining an `args_schema` with Pydantic ensures that the agent provides the correct arguments to your tool, reducing runtime errors.
*   **Clarity and Reusability:** This pattern makes the agent's capabilities and entry point clear and explicit, and the `SchedulingAgent` class can be easily reused.

This structure provides a robust foundation for building more complex and powerful CrewAI agents. Good luck!
