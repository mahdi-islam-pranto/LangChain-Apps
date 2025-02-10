from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

main_text_for_summary = """ 

কিছুক্ষণ আগে আপনার সাথে কথা হয়েছিল আপনার টিভি প্যানেল প্রবলেম হয়েছিল এবং আপনি বলেছেন যে আপনার শোরুম 
প্রোডাক্ট স্যার স্টক প্রোডাক্ট তাই তো স্টক প্রোডাক্ট না বলে শোরুমে আছে প্রোডাক্ট সেটা বলছি কাস্টমার প্রোডাক্ট কাস্টমার প্রোডাক্ট ধন্যবাদ 
স্যার সময় দিতে হচ্ছে থাকার জন্য স্যার একটি সার্ভিস রিকোর্ড রাখা হচ্ছে স্যার সময় দিবেন দিলার ফোন থেকে বলছি সিভিএন ইলেকট্রনিক 
ভুল করতেছেন বারবার। স্যার কান্ট্রি এড্রেসটি জানাবেন যেখানে আপনি সাপোর্টিং চাচ্ছেন। আপনি তো আমাকে আমার এড্রেস তো আর ইয়া 
সার্ভিস পয়েন্টে তো আর ইয়া আসে না টিভি টেকনিশিয়ান আসে না? আপনার আমাকে সার্ভিস পয়েন্টে দিয়া দিয়া আসতে হবে। আমি 
কাস্টমারের মোবাইল নাম্বার দেন। 01818 01818 84 846379 6379 নাম্বারটি কোনভাবে বসি মিলে দেখবেন 0818 846379 যেখানে 
সাপোর্ট চাচ্ছেন এড্রেসটি জানাবেন স্যার লোদের গাঁও জি স্যার লোদের গাঁও এল ও ডি ই আর জি ও ডব্লিউ লুদ্ধে যাও। ওকে স্যার। 
স্যার থানার নাম কারেন্টলি জানাবেন? সদর চাঁদপুর সদর চাঁদপুর ওকে স্যার। ধন্যবাদ স্যার একটি সার্ভিস রিকোয়ার্ট রাখা হয়ে তবে এই মুহূর্তে আপনার রেজিস্ট্রেশন কৃত ফোন নাম্বারে কোন সার্ভিস রিকোয়ার্ট কোড নাম্বার যাবে না কারণ আমাদের সিস্টেম আপডেট হচ্ছে তবে পরবর্তীতে আমাদের টেকনিশিয়ান টিম আপনার সাথে মেইন্টেন করে সার্ভিসটি প্রদান করে দিয়ে আসবে। স্যার আমার টিভি দেখা যায় আমার সার্ভিস ফোনটে পাঠিয়ে দিব। টিকিট নাম্বার লিখে পাঠিয়ে দিব। টিকিট নাম্বারটা স্যার টিকিট নাম্বারটি স্যার আপনার রেজিস্ট্রেশন কৃত ফোন নাম্বারে এই মুহূর্তে যাবে না কারণ আমাদের সিস্টেমটি আপডেট হচ্ছে কিন্তু আপনার যেহেতু রিকোয়েস্ট রাখা হয়ে সেক্ষেত্রে প্রপার্টীটি সার্ভিস পেয়ে যাবেন স্যার সঠিক টাইমে। ঠিক আছে স্যার আমাকে তো এই টিভিটা পাঠাইতে হয় সার্ভিস পয়েন্টে। আমি দুইটা টিভি যাবে দেখা যাবে একটা টিভি থেকে টাকা নেওয়া এখন টিকিট মানে কিভাবে এটা কি করবো? একটা টিকিট নাম্বার তো দিতে হয় সাথে। ক্ষেত্রে স্যার আপনার জোনাল স্যার অথবা টিওসি স্যারের সাথে যোগাযোগ করতে পারেন স্যার। আচ্ছা ঠিক আছে। ধন্যবাদ স্যার। আর কোনভাবে সহযোগিতা করতে পারছি কিনা কারেন্টলি জানা দেন। কাস্টমারের নাম লিখছেন? স্যার বিষয়টি জন্য দুঃখিত স্যার। কার্ডলি কাস্টমারের নামটি জানাবেন। এমদাদ হোসাইন জি স্যার এমদাদ হোসাইন ইমরান হোসাইন ডাইট স্যার এমদাদ এমদাদ হোসাইন এমদাদ এমদাদ ওকে স্যার মানে এটা কি নতুন ইয়ে করছেন Okay, sir. Okay, sir. এটা আপনি যে এটা আপনি এসাইন করবেন রিয়াঙ্কা ইলেকট্রনিক এন্ড ইলেকট্রনিক্স স্যার প্রিয়াঙ্কা রিয়াঙ্কা রিয়াঙ্কা ইলেকট্রিক এন্ড ইলেকট্রনিক्स হাজির হইলো ওকে স্যার কাস্টমার ইনফরমেশন সেন্টার নাম্বার 1657.


  """

chat_template = ChatPromptTemplate([
    SystemMessage("You are a Summarizer expert AI assistant in Bangla language. Always summarize the text in Bangla."),
    HumanMessage(""" I will give you a bangla text. The text is a transcription of an call center bangla audio conversation. 
                 I want you to summarize the bangla text for me. In your summary, Define key points of the customer query and call center agents conversation.
                 Such as, গ্রাহকের সমস্যা, গ্রাহকের তথ্য, কল সেন্টার এজেন্টের প্রতিক্রিয়া, গ্রাহকের অতিরিক্ত প্রশ্ন, etc
            """),
    AIMessage("অবশ্যই, আমি আপনার বাংলা টেক্সটকে সংক্ষিপ্ত করে দেব এবং গ্রাহকের প্রশ্ন ও কল সেন্টার এজেন্টের কথোপকথনের মূল পয়েন্টগুলি সংজ্ঞায়িত করব। অনুগ্রহ করে আপনার বাংলা টেক্সটটি আমাকে প্রদান করুন, আমি তা সংক্ষিপ্ত করে দেব।"),
    HumanMessage(main_text_for_summary)
])

# create the chat model
llm = ChatOpenAI(
    model="gpt-4",
)

format_chat_template = chat_template.format_messages()
print(f'full formated prompt: {format_chat_template}')


print("generating the summary...")
# generate the answer
response = llm.invoke(format_chat_template)
print("\n--- Answer ---")
print(response.content)
# save the response to a file with UTF-8 encoding
with open("SummaryAIresponse.txt", "w", encoding="utf-8") as f:
    f.write(response.content)
    