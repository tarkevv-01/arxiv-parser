# Система анализа научных публикаций

Микросервисная система для автоматического анализа и категоризации научных статей из arXiv с использованием LLM.

## Архитектура

Система состоит из двух микросервисов:

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ├─────────────────┐
       │                 │
       v                 v
┌──────────────┐  ┌──────────────┐
│   Fetcher    │  │   Analyzer   │
│   Service    │─>│   Service    │
│  (port 9000) │  │  (port 9001) │
└──────┬───────┘  └──────┬───────┘
       │                 │
       v                 v
┌──────────────┐  ┌──────────────┐
│  arXiv API   │  │ OpenRouter   │
│              │  │   LLM API    │
└──────────────┘  └──────────────┘
```

**Fetcher Service** — получает статьи из arXiv API, извлекает метаданные и опционально текст из PDF

**Analyzer Service** — анализирует статьи с помощью LLM, категоризирует и извлекает ключевую информацию

## 🚀 Быстрый старт

### 1. Настройка окружения

```bash
# Склонируйте репозиторий
git clone https://github.com/tarkevv-01/arxiv-parser
cd test_task

# Создайте .env файл
cp .env.example .env
```

Отредактируйте `.env` и добавьте ваш API ключ:
```env
OPENROUTER_API_KEY=your_api_key_here
```

> Получить бесплатный API ключ можно на [OpenRouter.ai](https://openrouter.ai/)

### 2. Запуск с Docker

```bash
docker-compose up --build
```

### 3. Проверка работоспособности

```bash
curl http://localhost:9000/health
curl http://localhost:9001/health
```

**Ожидаемый результат:**
```json
{"status":"healthy","service":"fetcher-service"}
{"status":"healthy","service":"analyzer-service","cache_size":0}
```

## Примеры использования CMD

### 1. Получение одной статьи по ID

**Команда:**
```bash
curl -X POST http://localhost:9000/fetch -H "Content-Type: application/json" -d "{\"arxiv_id\":\"2301.07041\",\"fetch_full_text\":false}"
```

**Ответ:**
```json
{
  "articles": [
    {
      "arxiv_id": "2301.07041",
      "title": "Verifiable Fully Homomorphic Encryption",
      "authors": ["Alexander Viand", "Christian Knabenhans", "Anwar Hithnawi"],
      "abstract": "Fully Homomorphic Encryption (FHE) is seeing increasing real-world deployment...",
      "categories": ["cs.CR"],
      "published": "2023-01-17T17:50:26Z",
      "pdf_url": "https://arxiv.org/pdf/2301.07041.pdf",
      "full_text": null,
      "text_length": null
    }
  ],
  "total": 1
}
```

### 2. Поиск статей по запросу

**Команда:**
```bash
curl -X POST http://localhost:9000/fetch -H "Content-Type: application/json" -d "{\"query\":\"machine learning\",\"max_results\":5,\"fetch_full_text\":false}"
```

**Ответ:**
```json
{
  "articles": [
    {
      "arxiv_id": "2306.04338",
      "title": "Changing Data Sources in the Age of Machine Learning for Official Statistics",
      "authors": ["Cedric De Boom", "Michael Reusens"],
      "abstract": "Data science has become increasingly essential...",
      "categories": ["stat.ML", "cs.LG"],
      "published": "2023-06-07T11:08:12Z",
      "pdf_url": "https://arxiv.org/pdf/2306.04338.pdf"
    },
    // ... еще 4 статьи
  ],
  "total": 5
}
```

### 3. Получение и анализ за один запрос

**Команда:**
```bash
curl -X POST http://localhost:9000/fetch-and-analyze -H "Content-Type: application/json" -d "{\"arxiv_id\":\"2301.07041\",\"fetch_full_text\":false}"
```

**Ответ:**
```json
{
  "articles": [
    {
      "arxiv_id": "2301.07041",
      "title": "Verifiable Fully Homomorphic Encryption",
      "authors": ["Alexander Viand", "Christian Knabenhans", "Anwar Hithnawi"],
      "abstract": "Fully Homomorphic Encryption (FHE) is seeing increasing real-world deployment...",
      "categories": ["cs.CR"],
      "published": "2023-01-17T17:50:26Z",
      "pdf_url": "https://arxiv.org/pdf/2301.07041.pdf"
    }
  ],
  "analysis": {
    "results": [
      {
        "arxiv_id": "2301.07041",
        "analysis": {
          "main_topic": "Enhancing Fully Homomorphic Encryption (FHE) with integrity to prevent malicious server attacks",
          "methodology": "Analysis of existing FHE integrity approaches, presentation of novel attacks, and proposal of a verifiable FHE framework",
          "key_findings": [
            "Existing FHE schemes lack integrity guarantees, enabling malicious servers to perform key-recovery attacks",
            "Prior work on FHE integrity is insufficient for modern deployment scenarios",
            "A new maliciously-secure verifiable FHE notion is proposed and instantiated with multiple techniques"
          ],
          "techniques": [
            "Verifiable FHE framework",
            "Key-recovery attack modeling",
            "Integrity-preserving computation protocols"
          ],
          "category": {
            "domain": "Computer Science",
            "subcategory": "Cryptography",
            "complexity": "Advanced",
            "article_type": "Theory"
          },
          "summary": {
            "brief": "The article addresses FHE's integrity vulnerabilities and proposes a verifiable framework to mitigate malicious server attacks through novel cryptographic techniques.",
            "key_points": [
              "FHE's malleability creates security risks when servers are untrusted",
              "Existing integrity solutions are fragmented and inadequate",
              "A comprehensive verifiable FHE approach is evaluated across diverse scenarios"
            ]
          }
        },
        "confidence": 0.85,
        "analysis_timestamp": "2026-01-29T03:43:41.136161Z"
      }
    ],
    "total": 1,
    "successful": 1,
    "failed": 0
  }
}
```

### 4. Анализ конкретной статьи

**Команда:**
```bash
curl -X POST http://localhost:9001/analyze -H "Content-Type: application/json" -d "{\"arxiv_id\":\"2301.07041\",\"title\":\"Verifiable Fully Homomorphic Encryption\",\"abstract\":\"Fully Homomorphic Encryption (FHE) is seeing increasing real-world deployment to protect data in use by allowing computation over encrypted data...\",\"categories\":[\"cs.CR\"]}"
```

**Ответ:**
```json
{
  "arxiv_id": "2301.07041",
  "analysis": {
    "main_topic": "Enhancing Fully Homomorphic Encryption (FHE) with integrity to prevent malicious server attacks",
    "methodology": "Analysis of existing FHE integrity approaches, presentation of novel attacks, and proposal of a verifiable FHE framework",
    "key_findings": [
      "Existing FHE schemes lack integrity guarantees, enabling malicious servers to perform key-recovery attacks",
      "Prior work on FHE integrity is insufficient for modern deployment scenarios",
      "A new maliciously-secure verifiable FHE notion is proposed and instantiated with multiple techniques"
    ],
    "techniques": [
      "Verifiable FHE framework",
      "Key-recovery attack modeling",
      "Integrity-preserving computation protocols"
    ],
    "category": {
      "domain": "Computer Science",
      "subcategory": "Cryptography",
      "complexity": "Advanced",
      "article_type": "Theory"
    },
    "summary": {
      "brief": "The article addresses FHE's integrity vulnerabilities and proposes a verifiable framework to mitigate malicious server attacks through novel cryptographic techniques.",
      "key_points": [
        "FHE's malleability creates security risks when servers are untrusted",
        "Existing integrity solutions are fragmented and inadequate",
        "A comprehensive verifiable FHE approach is evaluated across diverse scenarios"
      ]
    }
  },
  "confidence": 0.85,
  "analysis_timestamp": "2026-01-29T03:49:38.847235Z"
}
```

## API Endpoints

### Fetcher Service (localhost:9000)

| Endpoint | Method | Описание |
|----------|--------|----------|
| `/fetch` | POST | Получение статей по ID или поисковому запросу |
| `/fetch-and-analyze` | POST | Получение статей и их автоматический анализ |
| `/health` | GET | Проверка работоспособности сервиса |

### Analyzer Service (localhost:9001)

| Endpoint | Method | Описание |
|----------|--------|----------|
| `/analyze` | POST | Анализ одной статьи с помощью LLM |
| `/batch-analyze` | POST | Пакетный анализ нескольких статей |
| `/health` | GET | Проверка работоспособности сервиса |
| `/cache` | DELETE | Очистка кэша анализа |


## Безопасность

- Не коммитьте `.env` файл с реальными API ключами
- Используйте `.env.example` как шаблон
- API ключи передаются в контейнеры через переменные окружения
