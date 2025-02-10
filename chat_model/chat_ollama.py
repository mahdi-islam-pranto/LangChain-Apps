# run llama 3.1 using ollama 

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

chat_promt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a Summarizer expert AI assistant in Bangla language."),
    ("human", 
     """I will give you a bangla text. The text is a transcription of an call center bangla audio conversation. 
     I want you to summarize the text for me in Bangla and English both.
    .Give me the summary of this bangla text: 
        শেখ হাসিনার কার্যক্রমে ঢাকার প্রতিবাদ, নয়াদিল্লিকে নোট

ছবি: টেলিভিশন থেকে নেওয়া
গণঅভ্যুত্থানে ক্ষমতাচ্যুত প্রধানমন্ত্রী শেখ হাসিনার 'মিথ্যা ও বানোয়াট বক্তব্যের' তীব্র প্রতিবাদ জানিয়েছে বাংলাদেশ।


আজ বৃহস্পতিবার পররাষ্ট্র উপদেষ্টা তৌহিদ হোসেন গণমাধ্যমকে এই তথ্য জানিয়েছেন।

তিনি বলেন, 'বাংলাদেশকে অস্থিতিশীলতার দিকে ঠেলে দিতে সামাজিক যোগাযোগমাধ্যমসহ বিভিন্ন প্ল্যাটফর্মে অব্যাহতভাবে মিথ্যা ও বানোয়াট মন্তব্য এবং বিবৃতি দিয়ে যাচ্ছেন। পররাষ্ট্র মন্ত্রণালয় ভারতের কাছে এর প্রতিবাদ জানিয়েছে।'

সামাজিক যোগাযোগমাধ্যমে গত রাতে ভাষণ দেন শেখ হাসিনা। এর জের ধরে ধানমন্ডিতে বঙ্গবন্ধু স্মৃতি জাদুঘর এবং দেশের বিভিন্ন স্থানে বঙ্গবন্ধু শেখ মুজিবুর রহমানের ম্যুরাল ভেঙে ফেলছে বিক্ষুব্ধ জনতা।

ঢাকায় ভারতের ভারপ্রাপ্ত হাইকমিশনারের কাছে প্রতিবাদলিপি হস্তান্তর করা হয়েছে। এতে উল্লেখ করা হয়েছে, মন্ত্রণালয় গভীর উদ্বেগ, হতাশা ও গুরুতর আপত্তি জানাচ্ছে। কারণ এই ধরনের বক্তব্য বাংলাদেশের জনগণের অনুভূতিতে আঘাত করছে।


মন্ত্রণালয় আরও জোর দিয়ে বলেছে, তার (শেখ হাসিনা) এই ধরনের কর্মকাণ্ড বাংলাদেশের প্রতি শত্রুতামূলক কাজ হিসেবে বিবেচিত হচ্ছে এবং এটি দুই দেশের মধ্যে সুস্থ সম্পর্ক স্থাপনের প্রচেষ্টার জন্য সহায়ক নয়।

সোশ্যাল মিডিয়া এবং অন্যান্য যোগাযোগমাধ্যম ব্যবহার করে এই ধরনের মিথ্যা, বানোয়াট এবং উসকানিমূলক বিবৃতি দেওয়া থেকে শেখ হাসিনাকে বিরত রাখতে পারস্পরিক শ্রদ্ধা ও বোঝাপড়ার মনোভাব নিয়ে অবিলম্বে যথাযথ ব্যবস্থা নিতে ভারত সরকারকে অনুরোধ জানিয়েছে বাংলাদেশ।




        """
     
     ),
    
])

# create the chat model
llm = ChatOllama(
    model="llama3.1",
)

format_chat_promt_template = chat_promt_template.format_messages()
# generate the answer
response = llm.invoke(format_chat_promt_template)

print("\n--- Answer ---")
print(response.content)


# write the response to a file with UTF-8 encoding
with open("AIresponse.txt", "w", encoding="utf-8") as f:
    f.write(response.content)

print("AI response saved to AIresponse.txt")


