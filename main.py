from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import anthropic
import os
from dotenv import load_dotenv
import json
import chromadb
from chromadb.utils import embedding_functions
import uuid

load_dotenv()

app = FastAPI()

# CORS setup (For React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Anthropic client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ChromaDB setup (vector storage)
chroma_client = chromadb.Client()
embedding_function = embedding_functions.DefaultEmbeddingFunction()

# In-memory storage (to use database in production)
meetings_db = {}
quiz_db = {}


class QuizAnswer(BaseModel):
    question_id: int
    selected_option: int


class QuizSubmission(BaseModel):
    meeting_id: str
    player_name: str
    answers: List[QuizAnswer]


@app.get("/")
def read_root():
    return {"message": "Meeting Game API is running!"}


@app.post("/upload-transcript")
async def upload_transcript(file: UploadFile = File(...)):
    """
    Step 1: Upload meeting transcript
    Learning: File handling, async operations
    """
    
    # File validation
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files allowed")
    
    # Read transcript
    content = await file.read()
    transcript_text = content.decode('utf-8')
    
    if len(transcript_text) < 100:
        raise HTTPException(status_code=400, detail="Transcript too short")
    
    # Generate unique meeting ID
    meeting_id = str(uuid.uuid4())
    
    # Store transcript
    meetings_db[meeting_id] = {
        "transcript": transcript_text,
        "status": "processing"
    }
    
    # Process transcript (analyze + generate quiz)
    try:
        # STEP 1: Extract key information using Claude
        analysis = analyze_transcript(transcript_text)
        
        # STEP 2: Create vector embeddings for RAG
        store_embeddings(meeting_id, transcript_text, analysis)
        
        # STEP 3: Generate quiz questions
        quiz = generate_quiz(meeting_id, transcript_text, analysis)
        
        # Update database
        meetings_db[meeting_id].update({
            "status": "completed",
            "analysis": analysis,
            "quiz": quiz
        })
        
        quiz_db[meeting_id] = {
            "quiz": quiz,
            "leaderboard": []
        }
        
        return {
            "meeting_id": meeting_id,
            "status": "success",
            "analysis": analysis,
            "quiz_id": meeting_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


def analyze_transcript(transcript: str) -> Dict:
    """
    Learning: Prompt Engineering, Structured Output, Long-context handling
    """
    
    prompt = f"""Analyze this meeting transcript and extract:

1. Meeting duration (estimate from content)
2. Participants (list of names mentioned)
3. Key decisions made (with who proposed them)
4. Action items (with assignees)
5. Topics discussed
6. Topics mentioned but not fully discussed (ignored topics)

Transcript:
{transcript}

Return ONLY valid JSON in this exact format:
{{
  "duration_minutes": 45,
  "participants": ["Name1", "Name2"],
  "key_decisions": [
    {{"decision": "API timeout set to 30 seconds", "proposed_by": "Ram"}}
  ],
  "action_items": [
    {{"item": "Deploy by Friday", "assignee": "Priya"}}
  ],
  "topics_discussed": ["API design", "Deployment"],
  "ignored_topics": ["Security audit"]
}}

JSON:"""

    # Call Claude with structured output
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    response_text = message.content[0].text
    
    # Parse JSON (with guardrails)
    try:
        analysis = json.loads(response_text)
        
        # Validation (hallucination control)
        required_keys = ["duration_minutes", "participants", "key_decisions", 
                        "action_items", "topics_discussed", "ignored_topics"]
        
        for key in required_keys:
            if key not in analysis:
                analysis[key] = [] if key != "duration_minutes" else 30
                
        return analysis
        
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "duration_minutes": 30,
            "participants": [],
            "key_decisions": [],
            "action_items": [],
            "topics_discussed": [],
            "ignored_topics": []
        }


def store_embeddings(meeting_id: str, transcript: str, analysis: Dict):
    """
    Learning: Vector embeddings, RAG basics
    """
    
    # Create collection for this meeting
    collection = chroma_client.get_or_create_collection(
        name=f"meeting_{meeting_id}",
        embedding_function=embedding_function
    )
    
    # Split transcript into chunks (basic chunking strategy)
    chunks = []
    words = transcript.split()
    chunk_size = 200
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    
    # Store chunks with embeddings
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"chunk_index": i} for i in range(len(chunks))]
    )
    
    # Store analysis highlights separately
    highlights = []
    for decision in analysis.get("key_decisions", []):
        highlights.append(f"Decision: {decision['decision']} (by {decision['proposed_by']})")
    
    for item in analysis.get("action_items", []):
        highlights.append(f"Action: {item['item']} (assignee: {item['assignee']})")
    
    if highlights:
        collection.add(
            documents=highlights,
            ids=[f"highlight_{i}" for i in range(len(highlights))],
            metadatas=[{"type": "highlight"} for _ in highlights]
        )


def generate_quiz(meeting_id: str, transcript: str, analysis: Dict) -> Dict:
    """
    Learning: Advanced prompt engineering, Quiz generation
    """
    
    # Use RAG to get relevant context
    collection = chroma_client.get_collection(f"meeting_{meeting_id}")
    
    prompt = f"""Create a 7-question multiple choice quiz to test if someone actually attended this meeting.

Meeting Analysis:
{json.dumps(analysis, indent=2)}

Rules:
1. Questions should test actual engagement (not just skimming transcript)
2. Include questions about:
   - Specific decisions and who proposed them
   - Action items and assignees
   - Topics discussed vs ignored
   - Specific details that require active listening
3. Each question has 4 options (1 correct)
4. Make wrong options plausible but clearly wrong if you were present

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "id": 1,
      "question": "What timeout value did Ram propose for the API?",
      "options": ["10 seconds", "30 seconds", "60 seconds", "Never discussed"],
      "correct_answer": 1,
      "context": "Ram discussed API optimization"
    }}
  ]
}}

JSON:"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=3000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    try:
        quiz_data = json.loads(message.content[0].text)
        
        # Ensure we have exactly 7 questions
        questions = quiz_data.get("questions", [])[:7]
        
        # Validation
        for q in questions:
            if "id" not in q:
                q["id"] = questions.index(q) + 1
            if "correct_answer" not in q:
                q["correct_answer"] = 0
                
        return {"questions": questions}
        
    except:
        # Fallback quiz
        return {
            "questions": [
                {
                    "id": 1,
                    "question": "Was this meeting productive?",
                    "options": ["Yes", "No", "Maybe", "Can't remember"],
                    "correct_answer": 0
                }
            ]
        }


@app.get("/quiz/{meeting_id}")
def get_quiz(meeting_id: str):
    """Get quiz for a meeting"""
    if meeting_id not in quiz_db:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    return quiz_db[meeting_id]["quiz"]


@app.post("/submit-quiz")
def submit_quiz(submission: QuizSubmission):
    """
    Learning: Score calculation, Leaderboard logic
    """
    
    meeting_id = submission.meeting_id
    
    if meeting_id not in quiz_db:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    quiz = quiz_db[meeting_id]["quiz"]
    total_questions = len(quiz["questions"])
    correct_answers = 0
    
    # Calculate score
    for answer in submission.answers:
        question = next((q for q in quiz["questions"] if q["id"] == answer.question_id), None)
        if question and question["correct_answer"] == answer.selected_option:
            correct_answers += 1
    
    score_percentage = (correct_answers / total_questions) * 100
    
    # Assign badge
    if score_percentage >= 90:
        badge = "Actually Engaged 🏆"
    elif score_percentage >= 60:
        badge = "Mostly Present 👍"
    elif score_percentage >= 30:
        badge = "Physically There 😴"
    else:
        badge = "Who Are You Again? 👻"
    
    # Add to leaderboard
    player_entry = {
        "name": submission.player_name,
        "score": score_percentage,
        "badge": badge,
        "timestamp": str(uuid.uuid4())  # In production, use actual timestamp
    }
    
    quiz_db[meeting_id]["leaderboard"].append(player_entry)
    
    # Sort leaderboard
    quiz_db[meeting_id]["leaderboard"].sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "score": score_percentage,
        "correct": correct_answers,
        "total": total_questions,
        "badge": badge
    }


@app.get("/leaderboard/{meeting_id}")
def get_leaderboard(meeting_id: str):
    """Get leaderboard for a meeting"""
    if meeting_id not in quiz_db:
        raise HTTPException(status_code=404, detail="Leaderboard not found")
    
    return {
        "leaderboard": quiz_db[meeting_id]["leaderboard"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)