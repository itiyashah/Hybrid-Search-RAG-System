# Hybrid-Search-RAG-System
Combining semantic search + key-word search approach to find info from your local docs easily.
so in this readme i am going to tell you how to run this project locally in your own device 
1. you need to clone my repo : git clone <repository-url> ( copy the url under https tab or ssh tab if you have your ssh keys set up)
2. ensure the folder structure is same like this in your computer:
SMART-AI-ASSISTANT/
```text
├── app/
│   ├── static/
│   │   └── index.html
│   └── main.py
├── data/
│   └── (Put your PDF files here)
├── .env
├── .gitignore
└── requirements.txt
```

4. install all the dependencies listed in requirements.txt using the command: pip install -r requirements.txt
5. create .env file which will have your api key which does not need to be share with others , i have taken google api key from google ai studio as its free to use and this api key uses the latest gemini model : gemini-3.7-flash
6. i have added chroma_db/ in .gitignore file so that my saved database is not uploaded in GitHub and you can create your own fresh new database of your pdfs
7. you have to run the code(main.py) using following command : uvicorn app.main:app --reload 
8. Also ensure that you have set up your virtual environment :
python -m venv venv (u can use python3 depending on your version)
### On Windows:
venv\Scripts\activate
### On Mac/Linux:
source venv/bin/activate


 In terminal you'll get url something like this : 
     http://127.0.0.1:8000 
so click and open it

I was going to create a public link for this , but instead i thought let you'all clone my repo and download stuffs so you can actually know what happens under the hood and also you can modify or change the code as per your wish.
I just created this to understand my knowledge on classic rag system , how does it work , best way to learn any topic is to create a project on it.
Also one more important thing , you can upload the documents between the range of 10 to 20 inside the data folder and you can ask max 3 questions at a time to the agent related to different pdfs.
The agent uses conversational memory so it will remember your past conversations.
