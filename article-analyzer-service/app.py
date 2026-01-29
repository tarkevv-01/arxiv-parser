from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import json
from datetime import datetime
import asyncio
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Article Analyzer Service")

# Инициализация клиента OpenAI
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Кэш для результатов анализа
analysis_cache = {}

class ArticleInput(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    full_text: Optional[str] = None
    categories: List[str] = []

class CategoryInfo(BaseModel):
    domain: str
    subcategory: str
    complexity: str
    article_type: str

class SummaryInfo(BaseModel):
    brief: str
    key_points: List[str]

class AnalysisResult(BaseModel):
    main_topic: str
    methodology: Optional[str] = None
    key_findings: List[str]
    techniques: List[str]
    category: CategoryInfo
    summary: SummaryInfo

class AnalyzeResponse(BaseModel):
    arxiv_id: str
    analysis: AnalysisResult
    confidence: float
    analysis_timestamp: str

class BatchAnalyzeRequest(BaseModel):
    articles: List[ArticleInput]
    max_concurrent: int = 3

class BatchAnalyzeResponse(BaseModel):
    results: List[AnalyzeResponse]
    total: int
    successful: int
    failed: int

def create_analysis_prompt(article: ArticleInput) -> str:
    """Создание структурированного промпта для анализа статьи"""
    
    text_to_analyze = f"""
Title: {article.title}

Abstract: {article.abstract}

Categories: {', '.join(article.categories)}
"""
    
    if article.full_text:
        # Ограничиваем полный текст до 50000 символов
        full_text_limited = article.full_text[:50000]
        text_to_analyze += f"\n\nFull Text (excerpt):\n{full_text_limited}"
    
    prompt = f"""You are an expert scientific article analyzer. Analyze the following research article and provide a structured analysis in JSON format.

Article to analyze:
{text_to_analyze}

Provide your analysis in the following JSON structure (respond ONLY with valid JSON, no markdown formatting):

{{
  "main_topic": "Brief description of the main research topic",
  "methodology": "Research methodology used (or null if not applicable)",
  "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
  "techniques": ["Technique 1", "Technique 2"],
  "category": {{
    "domain": "Main domain (e.g., Computer Science, Physics, Mathematics, Biology, etc.)",
    "subcategory": "Specific subcategory (e.g., Machine Learning, Natural Language Processing, Computer Vision, Quantum Physics, etc.)",
    "complexity": "Beginner, Intermediate, or Advanced",
    "article_type": "Theory, Application, Survey, or Tutorial"
  }},
  "summary": {{
    "brief": "A 2-3 sentence summary of the article",
    "key_points": ["Key point 1", "Key point 2", "Key point 3"]
  }},
  "confidence": 0.85
}}

Respond with ONLY the JSON object, no additional text or formatting."""

    return prompt

async def analyze_with_llm(article: ArticleInput) -> dict:
    """Анализ статьи с использованием LLM"""
    
    # Проверка кэша
    cache_key = article.arxiv_id
    if cache_key in analysis_cache:
        print(f"Using cached result for {cache_key}")
        return analysis_cache[cache_key]
    
    try:
        prompt = create_analysis_prompt(article)
        
        response = await client.chat.completions.create(
            model="nvidia/nemotron-nano-9b-v2:free",
            messages=[
                {"role": "system", "content": "You are a scientific article analysis expert. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content
        
        # Очистка ответа от markdown форматирования
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Парсинг JSON ответа
        analysis_data = json.loads(content)
        
        # Валидация структуры ответа
        required_keys = ["main_topic", "key_findings", "techniques", "category", "summary"]
        for key in required_keys:
            if key not in analysis_data:
                raise ValueError(f"Missing required key in LLM response: {key}")
        
        # Валидация category
        category_keys = ["domain", "subcategory", "complexity", "article_type"]
        for key in category_keys:
            if key not in analysis_data["category"]:
                raise ValueError(f"Missing required key in category: {key}")
        
        # Валидация summary
        summary_keys = ["brief", "key_points"]
        for key in summary_keys:
            if key not in analysis_data["summary"]:
                raise ValueError(f"Missing required key in summary: {key}")
        
        # Сохранение в кэш
        analysis_cache[cache_key] = analysis_data
        
        return analysis_data
    
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to parse LLM response as JSON: {str(e)}. Response: {content[:200]}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during LLM analysis: {str(e)}"
        )

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_article(article: ArticleInput):
    """Анализ одной научной статьи"""
    
    try:
        # Валидация входных данных
        if not article.title or not article.abstract:
            raise HTTPException(
                status_code=400,
                detail="Title and abstract are required"
            )
        
        # Выполнение анализа
        analysis_data = await analyze_with_llm(article)
        
        # Извлечение confidence из ответа или использование значения по умолчанию
        confidence = analysis_data.get("confidence", 0.85)
        
        # Формирование ответа
        result = AnalysisResult(
            main_topic=analysis_data["main_topic"],
            methodology=analysis_data.get("methodology"),
            key_findings=analysis_data["key_findings"],
            techniques=analysis_data["techniques"],
            category=CategoryInfo(**analysis_data["category"]),
            summary=SummaryInfo(**analysis_data["summary"])
        )
        
        return AnalyzeResponse(
            arxiv_id=article.arxiv_id,
            analysis=result,
            confidence=confidence,
            analysis_timestamp=datetime.utcnow().isoformat() + "Z"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during analysis: {str(e)}"
        )

@app.post("/batch-analyze", response_model=BatchAnalyzeResponse)
async def batch_analyze_articles(request: BatchAnalyzeRequest):
    """Пакетный анализ нескольких статей"""
    
    if not request.articles:
        raise HTTPException(
            status_code=400,
            detail="No articles provided for analysis"
        )
    
    if request.max_concurrent < 1:
        raise HTTPException(
            status_code=400,
            detail="max_concurrent must be at least 1"
        )
    
    results = []
    successful = 0
    failed = 0
    
    # Создаем семафор для ограничения параллельности
    semaphore = asyncio.Semaphore(request.max_concurrent)
    
    async def analyze_with_semaphore(article: ArticleInput):
        async with semaphore:
            try:
                return await analyze_article(article)
            except Exception as e:
                print(f"Error analyzing article {article.arxiv_id}: {str(e)}")
                return None
    
    # Запускаем анализ всех статей с ограничением параллельности
    tasks = [analyze_with_semaphore(article) for article in request.articles]
    analysis_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Обработка результатов
    for result in analysis_results:
        if result is not None and not isinstance(result, Exception):
            results.append(result)
            successful += 1
        else:
            failed += 1
    
    return BatchAnalyzeResponse(
        results=results,
        total=len(request.articles),
        successful=successful,
        failed=failed
    )

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    
    # Проверка наличия API ключа
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return {
            "status": "unhealthy",
            "service": "analyzer-service",
            "error": "OPENROUTER_API_KEY not configured"
        }
    
    return {
        "status": "healthy",
        "service": "analyzer-service",
        "cache_size": len(analysis_cache)
    }

@app.delete("/cache")
async def clear_cache():
    """Очистка кэша анализа"""
    cache_size = len(analysis_cache)
    analysis_cache.clear()
    return {
        "status": "success",
        "cleared_entries": cache_size
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)