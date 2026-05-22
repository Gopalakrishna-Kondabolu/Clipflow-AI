import os
import json
import asyncio
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
from bs4 import BeautifulSoup
from groq import Groq

load_dotenv()

app = FastAPI(title="ClipFlow AI API", description="Core backend pipeline for ClipFlow AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://client-pi-seven-94.vercel.app",
        "https://client-j0dda2jyc-gopalakrishna-kondabolus-projects.vercel.app",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "online", "application": "ClipFlow AI Backend Engine"}

# Initialize Groq Client (automatically uses GROQ_API_KEY from environment)
try:
    client = Groq()
except Exception as e:
    client = None
    print(f"Warning: Failed to initialize Groq. Check GROQ_API_KEY. Error: {e}")

class RepurposeRequest(BaseModel):
    url: Optional[str] = None
    raw_text: Optional[str] = None
    target_audience: str

class ReelHooksRequest(BaseModel):
    topic: str
    tone: str

class LinkedinCarouselRequest(BaseModel):
    concept: str
    slides_count: int

class ShortsOptimizerRequest(BaseModel):
    raw_text: str

async def scrape_content(url: str) -> str:
    try:
        async with httpx.AsyncClient(follow_redirects=True) as http_client:
            response = await http_client.get(url, timeout=15.0)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            # Extract text from common text-bearing elements
            elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            text_content = "\n".join([el.get_text(strip=True) for el in elements if el.get_text(strip=True)])
            # Clean up white spaces
            text_content = " ".join(text_content.split())
            # Truncate content to roughly 1,500-2,000 words (8000 chars)
            return text_content[:8000]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to scrape URL: {str(e)}")

@app.post("/api/repurpose")
async def repurpose_content(request: RepurposeRequest):
    if not request.url and not request.raw_text:
        raise HTTPException(status_code=400, detail="Must provide either url or raw_text")

    if not client:
         raise HTTPException(status_code=500, detail="Groq client not initialized. Check GROQ_API_KEY environment variable.")

    content = ""
    if request.url:
        content = await scrape_content(request.url)
    elif request.raw_text:
        content = request.raw_text

    if not content.strip():
        raise HTTPException(status_code=400, detail="No readable content found to process.")

    system_instruction = """
    You are an expert short-form video scriptwriter and viral content strategist.
    Your task is to take the provided B2B technical blog content or raw text and repurpose it into exactly 3 distinct, high-impact vertical video scripts (approx. 60 seconds each) tailored for the specified target audience.
    
    Each script must include:
    1. A highly engaging "Hook Optimization" line (to grab attention in the first 3 seconds).
    2. A sequence of script elements where each element contains:
       - 'timestamp': Expected time marker (e.g., "0:00 - 0:03").
       - 'b_roll_suggestion': Specific visual tracking or B-roll suggestion for the editor.
       - 'script_line': The actual spoken text.
       
    You must output JSON using exactly these lowercase keys: 'hook_optimization' and 'elements'. Do not use spaces, capital letters, or alternate field names like 'script_elements' or 'Hook Optimization'.
    Output the response strictly in JSON format exactly matching the schema requested.
    """

    user_prompt = f"Target Audience: {request.target_audience}\n\nSource Content:\n{content}"

    try:
        response = None
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                break
            except Exception as inner_e:
                error_msg = str(inner_e)
                if attempt == 0 and ("429" in error_msg or "rate_limit" in error_msg.lower() or "RESOURCE_EXHAUSTED" in error_msg.upper()):
                    await asyncio.sleep(10)
                    continue
                raise inner_e
                
        # Extract raw text content from Groq
        raw_content = response.choices[0].message.content
        
        # Convert string output directly to a clean Python dictionary
        parsed_json = json.loads(raw_content)
        
        return parsed_json

    except json.JSONDecodeError:
        return JSONResponse(
            status_code=500,
            content={"detail": "Failed to parse AI response as valid JSON. Please try again."}
        )
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate_limit" in error_msg.lower() or "RESOURCE_EXHAUSTED" in error_msg.upper():
            return JSONResponse(
                status_code=429, 
                content={"detail": "The server is currently processing high traffic. Please try again in a few seconds."}
            )
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server processing error: {error_msg}"}
        )

@app.post("/api/reel-hooks")
async def generate_reel_hooks(request: ReelHooksRequest):
    if not client:
         raise HTTPException(status_code=500, detail="Groq client not initialized. Check GROQ_API_KEY environment variable.")
    
    system_instruction = """
    Generate 5 hyper-viral opening hooks for Instagram Reels. 
    Return a strict JSON format exactly matching this schema: 
    {"hooks": [{"type": "Curiosity/Fear/Value", "text": "The hook phrase", "psychology": "Why this works"}]}
    """
    user_prompt = f"Topic: {request.topic}\nTone: {request.tone}"

    try:
        response = None
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                break
            except Exception as inner_e:
                error_msg = str(inner_e)
                if attempt == 0 and ("429" in error_msg or "rate_limit" in error_msg.lower() or "RESOURCE_EXHAUSTED" in error_msg.upper()):
                    await asyncio.sleep(10)
                    continue
                raise inner_e
                
        raw_content = response.choices[0].message.content
        return json.loads(raw_content)

    except json.JSONDecodeError:
        return JSONResponse(status_code=500, content={"detail": "Failed to parse AI response as valid JSON."})
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate_limit" in error_msg.lower() or "RESOURCE_EXHAUSTED" in error_msg.upper():
            return JSONResponse(status_code=429, content={"detail": "The server is processing high traffic. Please try again."})
        return JSONResponse(status_code=500, content={"detail": f"Internal server error: {error_msg}"})


@app.post("/api/linkedin-carousel")
async def generate_linkedin_carousel(request: LinkedinCarouselRequest):
    if not client:
         raise HTTPException(status_code=500, detail="Groq client not initialized. Check GROQ_API_KEY environment variable.")
    
    system_instruction = """
    Design a high-engagement B2B LinkedIn carousel structure. 
    Return a strict JSON format exactly matching this schema: 
    {"title": "Carousel Title", "slides": [{"slide_num": 1, "header": "Slide Heading", "content": "Short punchy text", "visual": "Graphic suggestion"}]}
    """
    user_prompt = f"Concept: {request.concept}\nNumber of Slides: {request.slides_count}"

    try:
        response = None
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                break
            except Exception as inner_e:
                error_msg = str(inner_e)
                if ... and ("429" in error_msg or "rate_limit" in error_msg.lower() or "RESOURCE_EXHAUSTED" in error_msg.upper()):
                    await asyncio.sleep(10)
                    continue
                raise inner_e
                
        raw_content = response.choices[0].message.content
        return json.loads(raw_content)

    except json.JSONDecodeError:
        return JSONResponse(status_code=500, content={"detail": "Failed to parse AI response as valid JSON."})
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate_limit" in error_msg.lower() or "RESOURCE_EXHAUSTED" in error_msg.upper():
            return JSONResponse(status_code=429, content={"detail": "The server is processing high traffic. Please try again."})
        return JSONResponse(status_code=500, content={"detail": f"Internal server error: {error_msg}"})


@app.post("/api/shorts-optimizer")
async def generate_shorts_optimizer(request: ShortsOptimizerRequest):
    if not client:
         raise HTTPException(status_code=500, detail="Groq client not initialized. Check GROQ_API_KEY environment variable.")
    
    system_instruction = """
    Take a script or topic and generate highly clickable optimization frameworks for YouTube Shorts. 
    Return a strict JSON format exactly matching this schema: 
    {"titles": ["Title 1", "Title 2", "Title 3"], "description": "SEO optimized summary string", "tags": ["tag1", "tag2", "tag3"]}
    """
    user_prompt = f"Raw Text/Script:\n{request.raw_text}"

    try:
        response = None
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                break
            except Exception as inner_e:
                error_msg = str(inner_e)
                if attempt == 0 and ("429" in error_msg or "rate_limit" in error_msg.lower() or "RESOURCE_EXHAUSTED" in error_msg.upper()):
                    await asyncio.sleep(10)
                    continue
                raise inner_e
                
        raw_content = response.choices[0].message.content
        return json.loads(raw_content)

    except json.JSONDecodeError:
        return JSONResponse(status_code=500, content={"detail": "Failed to parse AI response as valid JSON."})
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate_limit" in error_msg.lower() or "RESOURCE_EXHAUSTED" in error_msg.upper():
            return JSONResponse(status_code=429, content={"detail": "The server is processing high traffic. Please try again."})
        return JSONResponse(status_code=500, content={"detail": f"Internal server error: {error_msg}"})
