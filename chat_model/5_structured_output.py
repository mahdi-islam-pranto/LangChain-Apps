from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel , Field
from typing import List
from dotenv import load_dotenv
load_dotenv()

# Define the schema for the structured output
class MobileReview(BaseModel):
    phone_model: str = Field(title="Phone Model", description="Name and model of the phone")
    rating: float = Field(title="Rating", description="The rating of the phone")
    pros: List = Field(title="Pros", description="The pros of the phone")
    cons: List = Field(title="Cons", description="The cons of the phone")
    short_summary: str = Field(title="Short Summary", description="A short summary of the review")


review_template = """
    Just got my hands on the new Galaxy S21 and wow, this thing is slick! The screen is gorgeous,
    colors pop like crazy. Camera's insane too, especially at night - my Insta game's never been
    stronger. Battery life's solid, lasts me all day no problem.
    Not gonna lie though, it's pretty pricey. And what's with ditching the charger? C'mon Samsung.
    Also, still getting used to the new button layout, keep hitting Bixby by mistake.
    Overall, I'd say it's a solid 4 out of 5. Great phone, but a few annoying quirks keep it from
    being perfect. If you're due for an upgrade, definitely worth checking out!
    """

# reiew template with chat template
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a mobile phone reviewer."),
        ("human", "Review the {phone_model}. Provide a rating, pros, cons, and a short summary."),
    ]
)

# format the review template
main_prompt = chat_template.format_prompt(phone_model = "Samsung Galaxy S24")

# Create a ChatOpenAI model
llm_model = ChatOpenAI(model="gpt-4")

structured_llm = llm_model.with_structured_output(MobileReview)

# provide the review template or chat template to the llm 
response = structured_llm.invoke(main_prompt)

# full Output
print(response)

# Structured Output
# Print only phone model and cons
print(f"Phone Model: {response.phone_model}")
print("\nCons:")
for con in response.cons:
    print(f"- {con}")


