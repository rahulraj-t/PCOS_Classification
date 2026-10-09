
import os
from flask import Blueprint, jsonify, request
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

chat_bp = Blueprint("chat", __name__)

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None



if not api_key:
    raise RuntimeError(
        f"GROQ_API_KEY not found. Check your .env file at: {ENV_PATH}"
    )

# Choose a model that is currently available in your Groq account.
MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

SYSTEM_PROMPT = """
You are WomenSafe, an AI assistant for a PCOS ultrasound
screening and awareness project.

Your responsibilities:
1. Explain PCOS, its symptoms, possible causes, diagnosis,
   lifestyle management, treatment options, and fertility.
2. Use clear, simple, empathetic language.
3. Give general health education, not personal medical diagnoses.
4. Never claim that an ultrasound image or AI prediction alone
   confirms or rules out PCOS.
5. Explain that PCOS assessment may require medical history,
   symptoms, laboratory tests, and clinical evaluation.
6. Do not prescribe medicines or recommend changing a
   prescribed dose. Refer treatment decisions to a clinician.
7. If a user reports severe pain, heavy bleeding, fainting,
   immediate danger, or thoughts of self-harm, encourage
   urgent professional help or local emergency services.
8. If you are uncertain, say so instead of inventing facts.
9. Keep answers concise, useful, and easy to understand.
10. Remind users when appropriate that your responses are
    general information and not a substitute for medical care.
"""

@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({
            "reply": "Please type a question."
        }), 400

    if len(message) > 1000:
        return jsonify({
            "reply": "Please keep your question under 1000 characters."
        }), 400

    if client is None:
        return jsonify({
            "reply": "The AI assistant is not configured yet. Please check the Groq API key."
        }), 503

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            temperature=0.4,
            max_tokens=500
        )

        reply = response.choices[0].message.content

        return jsonify({
            "reply": reply or "Sorry, I could not generate a response."
        })

    except Exception as e:
        import traceback

        print("ERROR CALLING GROQ API:")
        traceback.print_exc()

        return jsonify({
            "reply": f"AI assistant error: {type(e).__name__}: {e}"
        }), 502