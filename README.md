# Dialogue Summarization (SAMSum)

[![tests](/actions/workflows/tests.yml/badge.svg?branch=master)](/actions/workflows/tests.yml)

[![codecov](https://codecov.io/gh/Bogachevv/MLOps_project/branch/master/graph/badge.svg)](https://codecov.io/gh/Bogachevv/MLOps_project)

## Краткое описание

Цель проекта — построить сервис, который по переписке в чате (dialogue) автоматически генерирует краткое резюме (summary). В качестве модели используется LLM (конкретная архитектура/чекпойнт будут выбраны позже); модель дообучается (fine-tuning) на датасете **SAMSum** --- наборе диалогов в стиле мессенджеров с вручную размеченными краткими резюме.

Источники:
- SAMSum paper (Gliwa et al., 2019): https://aclanthology.org/D19-5409/
- SAMSum on Hugging Face (splits): https://huggingface.co/datasets/knkarthick/samsum

---

## Метрики

Ниже зафиксированы метрики для первой версии проекта. Метрики делятся на:

- **Онлайн (production SLI/SLO):** Error Rate, Latency p95
- **Оффлайн (качество модели):** ROUGE-L, BERTScore-F1, SummaC, QA-based factual consistency (QAFactEval)

### 1) Доля неуспешных запросов (Error Rate)

**Назначение:** надёжность API

**Как считаем:**  
ErrorRate = (# запросов со статусом 5xx + таймауты + application-level failure) / (# всех запросов).  
К 5xx относим серверные ошибки HTTP (см. RFC 9110). Дополнительно учитываем таймауты и случаи, когда сервис вернул ответ, но он непригоден (например, пустое резюме / некорректная структура, если введём схему).

**Целевое значение (SLO):**

- ErrorRate ≤ **1.0%** (скользящее окно 24 часа)

---

### 2) Latency p95

**Назначение:** производительность / UX

**Как считаем:**  
Считаем длительность обработки запроса (end-to-end) и берём **95-й перцентиль**: значение задержки, меньше или равно которому выполняются 95% запросов (по выбранному окну агрегации). Перцентили предпочтительнее среднего, т.к. они отражают “хвосты” задержек.

**Целевое значение (SLO):**

- Latency p95 ≤ **800 ms** (скользящее окно 5 минут)

---

### 3) ROUGE-L

**Назначение:** похожесть на эталон (регресс-гарда)

**Как считаем:**  
Считаем ROUGE-L между сгенерированным summary и референсным summary на фиксированном оффлайн-наборе (SAMSum test). ROUGE-L основан на **LCS (Longest Common Subsequence)** между текстами.

**Целевое значение (порог релиза):**

- ROUGE-L ≥ **???** на SAMSum test

Ссылки:
- ROUGE / ROUGE-L (Lin, 2004): https://aclanthology.org/W04-1013.pdf

---

### 4) BERTScore-F1

**Назначение:** семантическая близость к эталону

**Как считаем:**  
Считаем BERTScore и фиксируем **F1**, используя контекстные эмбеддинги токенов (по стандартной реализации). Оффлайн-оценка проводится на SAMSum test.

**Целевое значение (порог релиза):**

- BERTScore-F1 ≥ **???** на SAMSum test

Ссылки:
- BERTScore paper (Zhang et al., 2019): https://arxiv.org/abs/1904.09675

---

### 5) SummaC

**Назначение:** фактологическая согласованность / анти-галлюцинации

**Как считаем:**  
Используем метрику семейства SummaC для оценки согласованности summary с исходным текстом (dialogue). SummaC использует NLI-подход и агрегирует сигнал на уровне документа (а не только отдельных предложений).

**Целевое значение (порог релиза):**

- SummaC ≥ **???** на SAMSum test

Ссылки:
- SummaC paper (Laban et al., 2022): https://aclanthology.org/2022.tacl-1.10/

---

### 6) QA-based factual consistency (QAFactEval)

**Назначение:** фактологическая согласованность (QA-проверка)

**Как считаем:**  
Используем QA-based метрику **QAFactEval**: метрика строит вопросы/ответы и оценивает, насколько факты, присутствующие в summary, подтверждаются исходным текстом. Оффлайн-оценка проводится на SAMSum test.

**Целевое значение (порог релиза):**

- QAFactEval ≥ **???** на SAMSum test

Ссылки:
- QAFactEval paper (Fabbri et al., 2022): https://aclanthology.org/2022.naacl-main.187/
- Official implementation: https://github.com/salesforce/QAFactEval

---

## Примечание по порогам

Пороговые значения для оффлайн-метрик (ROUGE-L / BERTScore / SummaC / QAFactEval) в первой версии используются как **gating criteria** и будут уточняться после выбора базовой LLM и построения baseline-результатов на SAMSum.
