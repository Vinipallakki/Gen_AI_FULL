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
                return (
                    "Invalid date range. The start date cannot be after the end date."
                )

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

    SUPPORTED_CONTENT_TYPES = ["text/plain"]

    def __init__(self):
        """Initializes the SchedulingAgent."""
        if os.getenv("GOOGLE_API_KEY"):
            self.llm = LLM(
                model="gemini/gemini-2.0-flash",
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
