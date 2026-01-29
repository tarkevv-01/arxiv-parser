# fetcher-service/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import httpx
import os
import feedparser

app = FastAPI(title="Article Fetcher Service")

class FetchRequest(BaseModel):
    arxiv_id: Optional[str] = None
    query: Optional[str] = None
    max_results: int = 5
    fetch_full_text: bool = False

class Article(BaseModel):
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: str
    pdf_url: str
    full_text: Optional[str] = None
    text_length: Optional[int] = None

class FetchResponse(BaseModel):
    articles: List[Article]
    total: int

@app.post("/fetch", response_model=FetchResponse)
async def fetch_articles(request: FetchRequest):
    """Получение статей из arXiv"""
    
    # ИСПРАВЛЕНО: используем HTTPS вместо HTTP
    base_url = "https://export.arxiv.org/api/query?"
    
    if request.arxiv_id:
        url = f"{base_url}id_list={request.arxiv_id}"
    elif request.query:
        url = f"{base_url}search_query={request.query}&max_results={request.max_results}"
    else:
        raise HTTPException(status_code=400, detail="Необходимо указать arxiv_id или query")
    
    try:
        # ИСПРАВЛЕНО: добавлен параметр follow_redirects=True
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
        
        # Парсим XML с помощью feedparser
        feed = feedparser.parse(response.text)
        
        if not feed.entries:
            raise HTTPException(status_code=404, detail="Статьи не найдены")
        
        articles = []
        for entry in feed.entries:
            arxiv_id = entry.id.split('/abs/')[-1].split('v')[0]
            authors = [author.name for author in entry.authors]
            categories = [tag.term for tag in entry.tags]
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            article_data = {
                "arxiv_id": arxiv_id,
                "title": entry.title.replace('\n', ' ').strip(),
                "authors": authors,
                "abstract": entry.summary.replace('\n', ' ').strip(),
                "categories": categories,
                "published": entry.published,
                "pdf_url": pdf_url,
            }
            
            if request.fetch_full_text:
                full_text = await extract_pdf_text(pdf_url)
                article_data["full_text"] = full_text
                article_data["text_length"] = len(full_text) if full_text else 0
            
            articles.append(Article(**article_data))
        
        return FetchResponse(articles=articles, total=len(articles))
    
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при запросе к arXiv: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {str(e)}")

async def extract_pdf_text(pdf_url: str, max_pages: int = 10) -> Optional[str]:
    """Извлечение текста из PDF"""
    try:
        import pymupdf
        
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(pdf_url)
            response.raise_for_status()
            
            doc = pymupdf.open(stream=response.content, filetype="pdf")
            
            text_parts = []
            for page_num in range(min(len(doc), max_pages)):
                page = doc[page_num]
                text_parts.append(page.get_text())
            
            doc.close()
            
            full_text = "\n".join(text_parts)
            max_chars = 50000
            if len(full_text) > max_chars:
                full_text = full_text[:max_chars] + "..."
            
            return full_text
    
    except Exception as e:
        print(f"Ошибка при извлечении текста из PDF: {e}")
        return None


@app.post("/fetch-and-analyze", response_model=dict)
async def fetch_and_analyze(request: FetchRequest):
    """Получение и автоматический анализ статей"""
    
    # Получаем статьи
    fetch_result = await fetch_articles(request)
    
    # Отправляем на анализ
    analyzer_url = os.getenv("ANALYZER_SERVICE_URL", "http://localhost:9001")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{analyzer_url}/batch-analyze",
                json={
                    "articles": [article.dict() for article in fetch_result.articles],
                    "max_concurrent": 3
                }
            )
            response.raise_for_status()
            analysis_result = response.json()
        
        return {
            "articles": fetch_result.articles,
            "analysis": analysis_result
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при анализе: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "fetcher-service"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)