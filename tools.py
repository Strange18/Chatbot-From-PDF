from typing import Optional
from datetime import date
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from dateutil import parser
import re
from phonenumbers import parse as parse_phone, is_valid_number
from email_validator import validate_email, EmailNotValidError
from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
import os
from datetime import datetime, date
from langchain.agents import initialize_agent, AgentType

load_dotenv()

email_sent = False

user_details = {
    "name": None,
    "email": None,
    "phonenumber": None,
    "date_of_appointment": None,
}


# to parse the date given in other formats
def parse_date(date_str):
    try:
        # useing dateutil.parser
        return parser.parse(date_str, fuzzy=True).date()
    except ValueError:
        match = re.search(
            r"(\d{1,2})\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s*(\d{4})?",
            date_str,
            re.IGNORECASE,
        )
        if match:
            day = int(match.group(1))
            month = match.group(2).capitalize()
            year = int(match.group(3)) if match.group(3) else date.today().year
            return parser.parse(f"{day} {month} {year}").date()
    return None


# if model fails to extract the email and phone number it is extracted using regex
def extract_fallback(chat):
    email = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", chat)
    phone = re.search(r"\b\d{10}\b", chat)
    return email.group(0) if email else None, phone.group(0) if phone else None


# prompt for extracting details
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ Extract name of the person, their email address and phone number, and the date and time they prefer for an appointment with the Thapathali Campus administration. 
            Additionally, capture the department or administrator they wish to meet with (if specified).
            Follow the schema for more.
            - name
            - email
            - phonenumber
            - date_of_appointment (use YYYY-MM-DD format).
            Examples:
            1. '20 November' -> 'YYYY-11-20'
            2. 'next Monday' -> Calculate the actual date.
            today's date is {today_date}
            If information cannot be extracted, return null for that attribute. """,
        ),
        ("human", "{text}"),
    ]
)


# format for extracting the data
class Person(BaseModel):
    """Information about a person who is having conversion with the bot."""

    name: Optional[str] = Field(default=None, description="The name of the person.")
    email: Optional[str] = Field(
        default=None, description="The email address of the person to contact"
    )
    phonenumber: Optional[str] = Field(
        default=None,
        description="Phone Number of the person used to contact the person",
    )
    date_of_appointment: Optional[date] = Field(
        default=None, description="Date of Appointment when people want to scheudle."
    )


def detail_extractor(chat, llm):
    # llm = ChatMistralAI(
    #         model="mistral-large-latest",
    #         api_key=os.getenv("MISTRAL_API_KEY"),
    #         temperature=0,
    #     )
    runnable = prompt | llm.with_structured_output(schema=Person)

    try:
        result = runnable.invoke(
            {"text": chat, "today_date": datetime.today().strftime("%Y-%m-%d")}
        )
        return result

    except Exception as e:
        print("Error extracting details:", e)
        return None


# validating the extracted phone number
def validate_phone(phone):
    try:
        parsed = parse_phone(phone, "NP")
        return is_valid_number(parsed)
    except:
        return False


# validating the extracted email
def validate_email_address(email):
    try:
        if email is not None:
            validate_email(email)
            return True
        else:
            pass
    except EmailNotValidError:
        return False


def extract_details(chat):
    llm = ChatMistralAI(
        model="mistral-large-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0,
    )

    global user_details, email_sent
    if (
        not user_details["name"]
        or not user_details["email"]
        or not user_details["phonenumber"]
        or not user_details["date_of_appointment"]
    ):
        llm_result = detail_extractor(chat, llm=llm)

        if not (llm_result.email and llm_result.phonenumber):
            email, phone = extract_fallback(chat)
            if not llm_result.email:
                llm_result.email = email
            if not llm_result.phonenumber:
                llm_result.phonenumber = phone

        if llm_result.phonenumber:
            llm_result.phonenumber = (
                llm_result.phonenumber
                if validate_phone(llm_result.phonenumber)
                else None
            )
        if llm_result.email:
            llm_result.email = (
                llm_result.email if validate_email_address(llm_result.email) else None
            )
        if llm_result.date_of_appointment and not isinstance(
            llm_result.date_of_appointment, date
        ):
            llm_result.date_of_appointment = parse_date(llm_result.date_of_appointment)

        # updating the name in the global variable to save the details
        if llm_result.name and not user_details["name"]:
            user_details["name"] = llm_result.name
        if llm_result.email and not user_details["email"]:
            user_details["email"] = llm_result.email
        if llm_result.phonenumber and not user_details["phonenumber"]:
            user_details["phonenumber"] = llm_result.phonenumber
        if llm_result.date_of_appointment and not user_details["date_of_appointment"]:
            user_details["date_of_appointment"] = llm_result.date_of_appointment

        # email is only sent when all data is extracted properly
        if (
            user_details["name"]
            and user_details["email"]
            and user_details["phonenumber"]
            and user_details["date_of_appointment"]
            and not email_sent
        ):
            email_sender(chat_history=chat, extracted_details=user_details)
            email_sent = True
        return user_details
    else:
        return user_details


def email_sender(chat_history=None, extracted_details=None):

    llm = ChatMistralAI(
        model="mistral-large-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0,
    )
    credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/"],
        client_secrets_file="credentials.json",
    )
    api_resource = build_resource_service(credentials=credentials)
    toolkit = GmailToolkit(api_resource=api_resource)
    tools = toolkit.get_tools()

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )

    prompt = (
        f"Send Email the person whose details are given: {extracted_details}."
        f"Inform them about the confirmation about the appointment."
        f"Thank them for the good conversation by understanding their chat history: {chat_history}."
        "Try to maintain the writting format of the email. Do not write everything in a single line."
        "Sent the email as it sent from the Thapathali Campus."
        "Donot include any placeholders if you don't have the information do not write it out"
        "Maintain a professional and friendly tone. Use words that are easy to understand."
        "Generate output in JSON format without any backticks (`)."
        """ format for email should be like follows
        Dear [Recepient Name],
        [Contents]\n\n
        
        Regards \n
        Thapathali Campus 
        """
    )

    agent.invoke(prompt)

    # print("Email may be sent!")
