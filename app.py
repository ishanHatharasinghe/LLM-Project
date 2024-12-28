import importlib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()


try:
    bot1 = importlib.import_module("bot1")
    bot2 = importlib.import_module("bot2")
    bot3 = importlib.import_module("bot3")
    bot4 = importlib.import_module("bot4")
except ImportError as e:
    raise ImportError(f"Error importing modules: {e}")


class Question(BaseModel):
    query: str


@app.post("/ask")
def ask_chatbots(question: Question):
    try:
        answer1 = bot1.load_pdf_and_answer(question.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chatbot1: {e}")

    try:
        validation_query_1 = f"Is the following answer correct? '{answer1}'"
        answer2 = bot2.load_pdf_and_answer(validation_query_1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chatbot2: {e}")

    if "correct" in answer2.lower():
        validation_result_1 = "Correct"
        conclusion_1 = "Chatbot1's answer is valid."
    else:
        validation_result_1 = "Wrong"
        conclusion_1 = f"Chatbot1's answer is incorrect because: {answer2}"

    try:
        validation_query_3 = f"Is the following answer correct? '{answer1}'"
        answer3 = bot3.load_pdf_and_answer(validation_query_3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chatbot3: {e}")

    if "correct" in answer3.lower():
        validation_result_2 = "Correct"
        conclusion_2 = "Chatbot2's answer is valid."
    else:
        validation_result_2 = "Wrong"
        conclusion_2 = f"Chatbot2's answer is incorrect because: {answer3}"

    try:
        validation_query_4 = f"Is the following answer correct? '{answer3}'"
        answer4 = bot4.load_pdf_and_answer(validation_query_4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chatbot4: {e}")

    if "correct" in answer4.lower():
        validation_result_3 = "Correct"
        conclusion_3 = "Chatbot3's answer is valid."
    else:
        validation_result_3 = "Wrong"
        conclusion_3 = f"Chatbot3's answer is incorrect because: {answer4}"

    return {
        "chatbot1 answer": answer1,
        "validation result 1": validation_result_1,
        "chatbot2 answer": answer2,
        "conclusion 1": conclusion_1,
        "validation result 2": validation_result_2,
        "chatbot3 answer": answer3,
        "conclusion 2": conclusion_2,
        "validation result 3": validation_result_3,
        "chatbot4 answer": answer4,
        "conclusion 3": conclusion_3
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
