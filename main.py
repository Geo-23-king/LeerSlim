# =====================================================
# IMPORTS
# =====================================================
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sqlalchemy import create_engine, Column, String, Integer, Date, Text,DateTime,func
from sqlalchemy.orm import sessionmaker, declarative_base

from datetime import date, timedelta,datetime
from openai import OpenAI
from itsdangerous import URLSafeTimedSerializer
from dotenv import load_dotenv
from fastapi import HTTPException

import uuid
import smtplib
from email.message import EmailMessage
import os
import hashlib
import urllib.parse
from PIL import Image
import pytesseract
import io
import base64
from typing import List, Dict, Any,cast


## RUN uvicorn main:app --reload

#### BIG OPTION IS THIS: Ek kan nie die antwoord gee nie, maar ek kan jou help dink”
# =====================================================
# ENV + APP SETUP
# =====================================================
load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
serializer = URLSafeTimedSerializer(SECRET_KEY)

MOCK_AI = False
TRIAL_LIMIT = 3
DAILY_LIMIT = 10

# =====================================================
# DATABASE SETUP
# =====================================================
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./leerslim.db"  # fallback for local dev
)

engine_args = {}

# SQLite specific
if DATABASE_URL.startswith("sqlite"):
    engine_args["connect_args"] = {"check_same_thread": False}
else:
    # PostgreSQL / production
    engine_args["pool_pre_ping"] = True
    engine_args["pool_size"] = 10
    engine_args["max_overflow"] = 20

engine = create_engine(DATABASE_URL, **engine_args)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# =====================================================
# DATABASE MODELS
# =====================================================
class Learner(Base):
    __tablename__ = "learners"

    id = Column(String, primary_key=True)
    parent_email = Column(String)
    learner_name = Column(String)
    grade = Column(String)
    pin = Column(String)
    created = Column(Date, default=date.today)


class TrialUsage(Base):
    __tablename__ = "trial_usage"

    learner_id = Column(String, primary_key=True)
    used = Column(Integer, default=0)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    learner_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)
    created = Column(DateTime, default=datetime.utcnow)


class Subscription(Base):
    __tablename__ = "subscriptions"

    learner_id = Column(String, primary_key=True)
    start_date = Column(Date)
    end_date = Column(Date)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    learner_id = Column(String)
    grade = Column(String)
    rating = Column(String)
    comment = Column(Text)
    created = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# =====================================================
# CONSTANTS
# =====================================================
PRIMARY_SUBJECTS = [
    "Afrikaans", "Engels", "Wiskunde",
    "Natuurwetenskappe", "Sosiale Wetenskappe",
    "EBW", "Tegnologie"
]

SENIOR_SUBJECTS = [
    "Afrikaans Huistaal",
    "Afrikaans Eerste Addisionele Taal",
    "Engels Huistaal",
    "Engels Eerste Addisionele Taal",
    "Wiskunde",
    "Wiskundige Geletterdheid",
    "Fisiese Wetenskappe",
    "Lewenswetenskappe",
    "Geografie",
    "Geskiedenis",
    "Besigheidstudies",
    "Ekonomie",
    "Rekeningkunde",
    "Ingenieurs Grafika Ontwerp"
]

CHAT_MODEL = "gpt-4.1-mini"
VISION_MODEL = "gpt-4.1"


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def send_magic_login_email(to_email: str, learner_id: str):
    token = serializer.dumps(
        {
            "learner_id": learner_id,
            "action": "magic-login"
        }
    )

    base_url = os.getenv("BASE_URL", "http://127.0.0.1:8000")
    magic_url = f"{base_url}/magic-login?token={token}"

    msg = EmailMessage()
    msg["Subject"] = "LeerSlim – Veilige aanmelding 🔐"
    msg["From"] = "LeerSlim <no-reply@leerslim.co.za>"
    msg["To"] = to_email

    msg.set_content(f"""
Jy het versoek om veilig by LeerSlim aan te meld.

Klik op die skakel hieronder (geldig vir 15 minute):

{magic_url}

As jy dit nie versoek het nie, ignoreer hierdie e-pos.

LeerSlim
""")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(
            os.getenv("EMAIL_USER"),
            os.getenv("EMAIL_PASS")
        )
        smtp.send_message(msg)

def analyze_image_with_vision(
    image: UploadFile,
    learner,
    subject: str
) -> str:
    # Read image bytes
    image_bytes = image.file.read()
    image.file.seek(0)

    # Encode image to base64
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Vision prompt (CAPS-aligned, teacher-style)
    vision_prompt = f"""
Jy is LeerSlim, ’n CAPS-gefokusde Suid-Afrikaanse onderwyser.

Reëls:
- Graad: {learner.grade}
- Vak: {subject}
- Taal: Afrikaans
- Moenie finale antwoorde gee nie
- Gebruik net wat sigbaar is
- Prys moeite
- Vra EEN rigtinggewende vraag
"""

    # Vision input payload
    input_payload: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": vision_prompt
                },
                {
                    "type": "input_image",
                    "image_base64": image_b64
                }
            ]
        }
    ]

    # OpenAI Vision request (cast fixes PyCharm warning)
    response = client.responses.create(
        model=VISION_MODEL,
        input=cast(Any, input_payload),
        temperature=0.35,
        max_output_tokens=260
    )

    return response.output_text.strip()

def hash_pin(pin: str) -> str:
    return hashlib.sha256(pin.encode()).hexdigest()

def verify_pin(pin: str, hashed_pin: str) -> bool:
    return hash_pin(pin) == hashed_pin


def classify_schoolwork(text: str) -> str:
    text = text.lower()

    math_markers = ["+", "-", "=", "÷", "×", "bereken", "antwoord", "x", "y"]
    language_markers = ["sin", "paragraaf", "woord", "verduidelik", "lees"]

    if any(m in text for m in math_markers):
        return "math"

    if any(l in text for l in language_markers):
        return "language"

    return "general"

def paper_upload_prompt() -> str:
    return (
        "\n\n✍️ **Indien jy verkies:**\n"
        "• Jy kan jou antwoord hier in die teksblokkie intik, **of**\n"
        "• dit rustig op papier neerskryf.\n\n"
        "As jy papier gebruik, neem ’n **duidelike foto** "
        "en laai dit hier op sodat ons dit saam kan nagaan."
    )


def is_text_too_unclear(text: str) -> bool:
    if not text:
        return True

    words = text.split()
    return len(words) < 5

def extract_text_from_image(upload: UploadFile) -> str:
    image_bytes = upload.file.read()
    upload.file.seek(0)

    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale

    # Improve handwriting visibility
    image = image.point(lambda x: 0 if x < 150 else 255, "1")

    text = pytesseract.image_to_string(
        image,
        lang="eng",
        config="--psm 6"
    )

    return text.strip()

def is_likely_handwritten(text: str) -> bool:
    if not text:
        return False

    text = text.lower().strip()

    # ❌ Reject obvious non-school content
    banned = [
        "camera", "portrait", "face", "photo",
        "instagram", "facebook", "snapchat",
        "copyright", "www", "http", ".com"
    ]

    if any(word in text for word in banned):
        return False

    # ❌ OCR noise (selfies often produce this)
    if len(text) < 40:
        return False

    # ✅ Require numbers + structure
    digit_count = sum(c.isdigit() for c in text)
    line_count = text.count("\n")

    if digit_count >= 3 and line_count >= 1:
        return True

    return False

def get_current_learner(request: Request):
    token = request.cookies.get("session")
    if not token:
        return None
    try:
        return serializer.loads(token)["learner_id"]
    except:
        return None

def is_admin(request: Request) -> bool:
    # Simple admin protection (you can improve later)
    return request.cookies.get("admin") == os.getenv("ADMIN_TOKEN", "leerslim-admin")


def generate_payfast_url(data: dict) -> str:
    passphrase = os.getenv("PAYFAST_PASSPHRASE")

    # 1. Sort data alphabetically
    sorted_data = dict(sorted(data.items()))

    # 2. URL encode
    query = urllib.parse.urlencode(sorted_data)

    # 3. Append passphrase ONLY if it exists
    if passphrase:
        query += f"&passphrase={urllib.parse.quote(passphrase)}"

    # 4. Generate MD5 signature (UTF-8 REQUIRED)
    signature = hashlib.md5(query.encode("utf-8")).hexdigest()

    # 5. Return PayFast redirect URL
    return f"https://www.payfast.co.za/eng/process?{query}&signature={signature}"


def has_active_subscription(db, learner_id: str) -> bool:
    sub = db.query(Subscription).filter_by(learner_id=learner_id).first()
    if not sub:
        return False

    today = date.today()
    return sub.start_date <= today <= sub.end_date


def send_welcome_email(to_email, learner_name, grade, end_date):
    msg = EmailMessage()
    msg["Subject"] = "Welkom by LeerSlim Pro 🎓"
    msg["From"] = "LeerSlim <no-reply@leerslim.co.za>"
    msg["To"] = to_email

    msg.set_content(f"""
Beste Ouer / Voog,

Welkom by LeerSlim Pro!

🎉 Jou betaling was suksesvol en toegang is nou aktief.

Leerder: {learner_name}
Graad: {grade}
Toegang geldig tot: {end_date}

Wat LeerSlim bied:
✔ CAPS-gebaseerde hulp
✔ Geen finale antwoorde – net begrip
✔ Veilige leeromgewing
✔ Tot 10 vrae per dag

Vriendelike groete  
LeerSlim Span
""")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(
            os.getenv("EMAIL_USER"),
            os.getenv("EMAIL_PASS")
        )
        smtp.send_message(msg)

def get_ai_response(
    question: str,
    subject: str,
    grade: str,
    help_style: str
) -> str:

    style_map = {
        "step": """
Verduidelik kortliks die konsep in jou eie woorde.
Breek dit in hoogstens 2–3 eenvoudige stappe op.
Vra daarna EEN duidelike opvolgvraag.
""",

        "example": """
Gee EERS een eenvoudige, graad-toepaslike voorbeeld.
Verduidelik net genoeg om begrip te bou.
Vra daarna die leerder om self ’n soortgelyke voorbeeld te probeer.
""",

        "test": """
Moenie verduidelik nie.
Vra net een of twee vrae om begrip te toets.
Wag vir die leerder se antwoord voordat jy verder gaan.
"""
    }

    system_prompt = f"""
Jy is LeerSlim, ’n professionele CAPS-gefokusde onderwyser vir Suid-Afrikaanse skole.

🎓 ONDERWYSER-BEGINSELS:
- Graad: {grade}
- Vak: {subject}
- Taal: Afrikaans
- Tree op soos ’n regte onderwyser in ’n klaskamer
- Bou begrip, nie antwoorde nie
- Moenie huiswerk namens die leerder doen nie

🧠 PEDAGOGIESE GEDRAG:
- Moenie vaste frases herhaal nie
- Moenie altyd dieselfde opening gebruik nie
- Gebruik natuurlike variasie soos ’n menslike onderwyser
- Prys die DENKPROSES, nie die finale antwoord nie
- Lei sagkens reg indien nodig
- Vra net EEN vraag op ’n slag

🚫 BELANGRIK:
- Sê “Ek kan nie die antwoord gee nie…” NET
  as die leerder DIREK vra vir ’n antwoord
- Andersins, lei sonder om dit te sê

🧩 AFSLUITING:
Gebruik een van hierdie, slegs indien dit natuurlik pas:
- “Skryf jou antwoord hieronder.”
- “Verduidelik hoe jy daaroor dink.”
- “Probeer dit self en sê vir my hoe jy dink.”
- “Kom ons werk dit stap vir stap saam deur.”

INSTRUKSIE VIR HIERDIE ANTWOORD:
{style_map.get(help_style, style_map["step"])}

Leerder se boodskap:
{question}
"""

    response = client.responses.create(
        model=CHAT_MODEL,
        input=system_prompt,
        temperature=0.35,
        max_output_tokens=260,
    )

    return response.output_text.strip()

# =====================================================
# ROUTES — PUBLIC
# =====================================================
@app.get("/", response_class=HTMLResponse)
def home():
    return open("templates/index.html", encoding="utf-8").read()


@app.get("/signup", response_class=HTMLResponse)
def signup(request: Request):
    return templates.TemplateResponse(
        "signup.html",
        {
            "request": request,
            "error": None
        }
    )



@app.get("/logout")
def logout():
    response = RedirectResponse("/signup", status_code=302)
    response.delete_cookie("session")
    return response


# =====================================================
# AUTH / SIGNUP
# =====================================================
@app.post("/signup")
def signup_or_login(
    request: Request,
    parent_email: str = Form(...),
    learner_name: str = Form(""),
    grade: str = Form(""),
    learner_pin: str = Form(...)
):
    db = SessionLocal()

    learner = db.query(Learner).filter_by(parent_email=parent_email).first()

    # -----------------------------
    # LOGIN FLOW (email exists)
    # -----------------------------
    if learner:
        if learner.pin != hash_pin(learner_pin):
            db.close()
            return templates.TemplateResponse(
                "signup.html",
                {
                    "request": request,
                    "error": "❌ Verkeerde PIN. Probeer asseblief weer."
                }
            )

        token = serializer.dumps({"learner_id": learner.id})
        response = RedirectResponse("/chat", status_code=302)
        response.set_cookie(
            "session",
            token,
            httponly=True,
            samesite="lax",
            secure=False  # change to True when HTTPS is live
        )
        db.close()
        return response

    # -----------------------------
    # SIGNUP FLOW (new email)
    # -----------------------------
    if not learner_name or not grade:
        db.close()
        return templates.TemplateResponse(
            "signup.html",
            {
                "request": request,
                "error": "⚠️ Vul asseblief die leerder se naam en graad in."
            }
        )

    learner_id = str(uuid.uuid4())

    db.add(Learner(
        id=learner_id,
        parent_email=parent_email,
        learner_name=learner_name,
        grade=grade,
        pin=hash_pin(learner_pin),
        created=date.today()
    ))
    db.commit()
    db.close()

    token = serializer.dumps({"learner_id": learner_id})
    response = RedirectResponse("/chat", status_code=302)
    response.set_cookie(
        "session",
        token,
        httponly=True,
        samesite="lax",
        secure=False  # change to True when HTTPS is live
    )
    return response

@app.get("/forgot-pin", response_class=HTMLResponse)
def forgot_pin(request: Request):
    return templates.TemplateResponse(
        "forgot_pin.html",
        {
            "request": request,
            "error": None
        }
    )

@app.post("/forgot-pin", response_class=HTMLResponse)
def forgot_pin_submit(
    request: Request,
    parent_email: str = Form(...)
):
    db = SessionLocal()
    learner = db.query(Learner).filter_by(parent_email=parent_email).first()

    if learner:
        try:
            send_magic_login_email(parent_email, learner.id)
        except Exception as e:
            print("Magic login email failed:", e)

    db.close()

    return templates.TemplateResponse(
        "forgot_pin.html",
        {
            "request": request,
            "message": "✅ As die e-pos bestaan, is ’n veilige aanmeld-skakel gestuur."
        }
    )

@app.get("/magic-login")
def magic_login(token: str):
    try:
        data = serializer.loads(token, max_age=900)  # 15 minutes
        if data.get("action") != "magic-login":
            raise ValueError()
    except:
        raise HTTPException(status_code=400, detail="Skakel is ongeldig of verval")

    response = RedirectResponse("/chat", status_code=302)
    session_token = serializer.dumps({"learner_id": data["learner_id"]})
    response.set_cookie(
        "session",
        session_token,
        httponly=True,
        samesite="lax",
        secure=False  # will change later
    )

    return response

# =====================================================
# CHAT UI
# =====================================================
@app.get("/chat", response_class=HTMLResponse)
def chat(request: Request):
    learner_id = get_current_learner(request)
    if not learner_id:
        return RedirectResponse("/signup")

    db = SessionLocal()
    learner = db.query(Learner).filter_by(id=learner_id).first()
    messages = db.query(ChatMessage)\
        .filter_by(learner_id=learner_id)\
        .order_by(ChatMessage.id).all()

    is_pro = has_active_subscription(db, learner_id)

    trial = db.query(TrialUsage).filter_by(learner_id=learner_id).first()
    remaining_free = (
        max(0, TRIAL_LIMIT - trial.used)
        if trial and not is_pro
        else 0
    )

    db.close()

    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "name": learner.learner_name,
            "grade": learner.grade,
            "messages": messages,
            "remaining_free": remaining_free,
            "is_pro": is_pro
        }
    )


# =====================================================
# CHAT SUBMISSION
# =====================================================

@app.post("/submit")
async def submit_homework(
    request: Request,
    question: str = Form(""),
    subject: str = Form(...),
    help_style: str = Form(...),
    image: UploadFile | None = File(None)
):
    learner_id = get_current_learner(request)
    if not learner_id:
        return JSONResponse({"error": "Session expired"}, status_code=403)

    db = SessionLocal()
    learner = db.query(Learner).filter_by(id=learner_id).first()

    # -------------------------------------------------
    # QUESTION LENGTH SAFETY CHECK
    # -------------------------------------------------
    if question and len(question) > 2000:
        db.close()
        return JSONResponse({
            "answer": "⚠️ Jou vraag is te lank. Probeer dit asseblief korter maak."
        })

    extracted_text = ""

    # =====================================================
    # IMAGE SCAN + VALIDATION (NO LIMIT DEDUCTION)
    # =====================================================
    if image:
        extracted_text = extract_text_from_image(image)

        if not is_likely_handwritten(extracted_text):
            db.close()
            return JSONResponse({
                "answer": (
                    "📸 Ek sien jy het ’n foto opgelaai.\n\n"
                    "Dit lyk egter nie soos **handgeskrewe skoolwerk** nie.\n\n"
                    "📘 Neem asseblief ’n duidelike foto van jou werk op papier "
                    "(reg van bo af, goeie lig).\n\n"
                    "🔁 Hierdie poging tel **nie** teen jou vrae nie."
                ),
                "remaining": None
            })

        if is_text_too_unclear(extracted_text):
            db.close()
            return JSONResponse({
                "answer": (
                    "📸 Ek sien jou werk 👍\n\n"
                    "Die skrif is ’n bietjie moeilik om te lees.\n\n"
                    "👉 Kan jy vir my sê wat die vraag vra\n"
                    "of wat jy in **stap 1** probeer doen het?"
                ),
                "remaining": None
            })

    # =====================================================
    # LIMIT CHECK
    # =====================================================
    if has_active_subscription(db, learner_id):
        used_today = db.query(ChatMessage).filter(
            ChatMessage.learner_id == learner_id,
            ChatMessage.role == "user",
            func.date(ChatMessage.created) == date.today()
        ).count()

        if used_today >= DAILY_LIMIT:
            db.close()
            return JSONResponse({
                "limit_reached": True,
                "message": "Jy het vandag se 10 vrae gebruik."
            })
    else:
        trial = db.query(TrialUsage).filter_by(learner_id=learner_id).first()
        if not trial:
            trial = TrialUsage(learner_id=learner_id, used=0)
            db.add(trial)
            db.commit()

        if trial.used >= TRIAL_LIMIT:
            db.close()
            return JSONResponse({
                "limit_reached": True,
                "message": "Jy het jou 3 gratis vrae gebruik."
            })

        trial.used += 1
        db.commit()

    # =====================================================
    # SUBJECT CHECK
    # =====================================================
    grade_num = int(str(learner.grade))
    allowed = PRIMARY_SUBJECTS if grade_num <= 9 else SENIOR_SUBJECTS

    if subject not in allowed:
        db.close()
        return JSONResponse({"answer": "⚠️ Vak nie beskikbaar nie."})

    # =====================================================
    # AI RESPONSE (VISUALLY GROUNDED)
    # =====================================================
    # =====================================================
    # AI RESPONSE (VISUALLY GROUNDED)
    # =====================================================
    try:
        if image:
            work_type = classify_schoolwork(extracted_text)

            ai_prompt = f"""
    A learner uploaded handwritten schoolwork.

    IMPORTANT:
    - Only use what is visible below
    - Do NOT assume missing steps
    - Do NOT give final answers
    - Act like a CAPS teacher

    Visible content:
    ----------------
    {extracted_text}
    ----------------

    Work type: {work_type}

    Teacher rules:
    - Acknowledge effort
    - Refer only to visible work
    - Ask ONE guiding question
    - Be calm and encouraging
    """
            answer = get_ai_response(
                question=ai_prompt,
                subject=subject,
                grade=str(learner.grade),
                help_style="step"
            )

            # ✅ PROFESSIONAL ACKNOWLEDGEMENT (IMAGE ONLY)
            answer = (
                    "📸 Ek sien jou werk 👍\n\n"
                    "Dankie dat jy dit op papier neergeskryf het.\n"
                    "Kom ons kyk nou saam daarna.\n\n"
                    + answer
            )

        else:
            answer = get_ai_response(
                question=question,
                subject=subject,
                grade=str(learner.grade),
                help_style=help_style
            )

            # ✅ Add choice ONLY when learner typed text
            answer += paper_upload_prompt()

    except Exception:
        answer = (
            "⚠️ Iets het verkeerd geloop.\n\n"
            "Ek kan jou nie nou help nie, maar jy kan dit weer probeer "
            "of jou werk duideliker stuur.\n\n"
            "Ek is hier om jou te help dink 💙"
        )
    # =====================================================
    # SAVE CHAT
    # =====================================================
    user_content = (
        "📸 Ek het my werk opgelaai."
        if image and not question.strip()
        else question
    )

    db.add(ChatMessage(
        learner_id=learner_id,
        role="user",
        content=user_content,
        created=datetime.utcnow()
    ))

    db.add(ChatMessage(
        learner_id=learner_id,
        role="assistant",
        content=answer,
        created=datetime.utcnow()
    ))

    db.commit()

    remaining = None
    if not has_active_subscription(db, learner_id):
        trial = db.query(TrialUsage).filter_by(learner_id=learner_id).first()
        remaining = max(0, TRIAL_LIMIT - trial.used)

    db.close()

    return JSONResponse({
        "answer": answer,
        "remaining": remaining
    })
# =====================================================
# UTILITIES
# =====================================================
@app.post("/clear-chat")
def clear_chat(request: Request):
    learner_id = get_current_learner(request)
    if not learner_id:
        return RedirectResponse("/signup")

    db = SessionLocal()
    db.query(ChatMessage)\
        .filter(ChatMessage.learner_id == learner_id)\
        .delete()
    db.commit()
    db.close()

    return RedirectResponse("/chat", status_code=302)

@app.post("/feedback")
async def submit_feedback(request: Request):
    data = await request.json()

    learner_id = get_current_learner(request)
    if not learner_id:
        raise HTTPException(status_code=403)

    db = SessionLocal()
    learner = db.query(Learner).filter_by(id=learner_id).first()

    fb = Feedback(
        learner_id=learner_id,
        grade=learner.grade,
        rating=data.get("rating"),
        comment=data.get("comment", "")
    )

    db.add(fb)
    db.commit()
    db.close()

    return {"status": "ok"}

@app.get("/admin/feedback", response_class=HTMLResponse)
def admin_feedback(request: Request):
    if not is_admin(request):
        raise HTTPException(status_code=403)

    db = SessionLocal()
    feedback = db.query(Feedback)\
        .order_by(Feedback.created.desc())\
        .all()
    db.close()

    return templates.TemplateResponse(
        "admin_feedback.html",
        {
            "request": request,
            "feedback": feedback
        }
    )

@app.get("/pay")
def pay(request: Request):
    learner_id = get_current_learner(request)
    if not learner_id:
        return RedirectResponse("/signup")

    db = SessionLocal()
    learner = db.query(Learner).filter_by(id=learner_id).first()
    db.close()

    data = {
        "merchant_id": os.getenv("PAYFAST_MERCHANT_ID"),
        "merchant_key": os.getenv("PAYFAST_MERCHANT_KEY"),

        "return_url": f"{os.getenv('BASE_URL')}/payment-success",
        "cancel_url": f"{os.getenv('BASE_URL')}/chat",
        "notify_url": f"{os.getenv('BASE_URL')}/payfast/notify",

        "name_first": learner.learner_name,
        "email_address": learner.parent_email,

        "m_payment_id": learner_id,
        "item_name": "LeerSlim Pro Subscription",

        "amount": "279.99",

        # Subscription
        "subscription_type": "1",
        "billing_date": (date.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "recurring_amount": "279.99",
        "frequency": "1",   # Monthly
        "cycles": "0"       # Until cancelled
    }

    url = generate_payfast_url(data)
    return RedirectResponse(url)

@app.get("/payment-success")
def payment_success(request: Request):
    learner_id = get_current_learner(request)
    if not learner_id:
        return RedirectResponse("/signup")

    db = SessionLocal()
    db.merge(Subscription(
        learner_id=learner_id,
        start_date=date.today(),
        end_date=date.today() + timedelta(days=30)
    ))
    db.commit()
    db.close()

    return RedirectResponse("/chat")


@app.post("/payfast/notify")
async def payfast_notify(request: Request):
    data = dict(await request.form())

    pf_signature = data.pop("signature", "")
    passphrase = os.getenv("PAYFAST_PASSPHRASE")

    # Rebuild signature string
    query = urllib.parse.urlencode(sorted(data.items()))

    if passphrase:
        query += f"&passphrase={urllib.parse.quote(passphrase)}"

    calculated_signature = hashlib.md5(query.encode("utf-8")).hexdigest()

    # Reject invalid signatures
    if calculated_signature != pf_signature:
        return JSONResponse({"error": "Invalid signature"}, status_code=400)

    # Process successful payment
    if data.get("payment_status") == "COMPLETE":
        learner_id = data.get("m_payment_id")

        db = SessionLocal()
        db.merge(Subscription(
            learner_id=learner_id,
            start_date=date.today(),
            end_date=date.today() + timedelta(days=30)
        ))
        db.commit()
        db.close()

    # PayFast REQUIRES 200 OK
    return JSONResponse({"status": "ok"}, status_code=200)