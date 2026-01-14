"""
Neuro-consultant for the Technical Regulation of the Customs Union
"On the Safety of Railway Rolling Stock".

This script is designed for Google Colab and includes required modules:
1) Install dependencies
2) API key (secure)
3) DOCX -> text
4) Chunking
5) Vector DB (FAISS)
6) Search
7) OpenAI (LLM)
8) Answer function
9) Checks (sample questions)
"""

# ---------------------------
# 1) Установка зависимостей
# ---------------------------
# В Colab выполните:
# !pip -q install --upgrade langchain langchain-community langchain-openai faiss-cpu python-docx tiktoken

# ---------------------------
# 2) Ключ API (безопасно)
# ---------------------------
import os
from getpass import getpass

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass("Введите OpenAI API Key: ")

# ---------------------------
# 3) Чтение DOCX → текст
# ---------------------------
from docx import Document


def load_docx_text(docx_path: str) -> str:
    doc = Document(docx_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


# ---------------------------
# 4) Разбиение на фрагменты (чанки)
# ---------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.create_documents([text])


# ---------------------------
# 5) Векторная база (FAISS)
# ---------------------------
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def build_vector_db(docs):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)


# ---------------------------
# 6) Поиск по регламенту
# ---------------------------

def search_with_scores(db, query: str, k: int = 6):
    return db.similarity_search_with_score(query, k=k)


# ---------------------------
# 7) Подключение OpenAI (LLM)
# ---------------------------
from langchain_openai import ChatOpenAI


def build_llm(model: str = "gpt-4o-mini", temperature: float = 0.2):
    return ChatOpenAI(model=model, temperature=temperature)


# ---------------------------
# 8) Функция ответа нейроконсультанта
# ---------------------------
SYSTEM_PROMPT = (
    "Ты — нейро-консультант по Техническому регламенту Таможенного союза "
    "«О безопасности железнодорожного подвижного состава». "
    "Отвечай только на основе предоставленных фрагментов документа. "
    "Если в документах нет ответа или вопрос не относится к регламенту, "
    "скажи: «В регламенте нет информации для ответа на этот вопрос». "
    "Отвечай на русском, кратко и по делу."
)


def answer_question(
    db,
    llm,
    question: str,
    k: int = 6,
    relevance_threshold: float = 0.35,
):
    """
    relevance_threshold: порог релевантности (меньше — ближе).
    Используется для фильтрации результатов similarity_search_with_score.
    """
    results = search_with_scores(db, question, k=k)

    # Все найденные чанки и их скоры (для вывода)
    all_scores = [(doc.page_content[:200], score) for doc, score in results]

    # Фильтрация по порогу
    filtered = [(doc, score) for doc, score in results if score <= relevance_threshold]

    if not filtered:
        return {
            "answer": "В регламенте нет информации для ответа на этот вопрос.",
            "all_scores": all_scores,
        }

    context = "\n\n".join(
        [f"Фрагмент (score={score:.4f}):\n{doc.page_content}" for doc, score in filtered]
    )

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Контекст документа:\n{context}\n\n"
        f"Вопрос пользователя: {question}\n"
        f"Ответ:"
    )

    response = llm.invoke(prompt)
    return {
        "answer": response.content.strip(),
        "all_scores": all_scores,
    }


# ---------------------------
# 9) Проверка (как требует задание)
# ---------------------------
if __name__ == "__main__":
    # Укажите путь к DOCX, загруженному в Colab (например, /content/регламент.docx)
    docx_path = "/content/technical_reglament_railway.docx"
    text = load_docx_text(docx_path)

    docs = chunk_text(text)
    db = build_vector_db(docs)
    llm = build_llm()

    test_questions = [
        "Какие требования предъявляются к тормозным системам подвижного состава?",
        "Разрешены ли одноразовые пластиковые бутылки в вагонах?",
        "Как регламент определяет проверку прочности кузова?",
    ]

    for question in test_questions:
        result = answer_question(db, llm, question)
        print("\nВопрос:", question)
        print("Ответ:", result["answer"])
        print("Скоры найденных чанков:")
        for snippet, score in result["all_scores"]:
            print(f"  score={score:.4f} | {snippet.replace('\n', ' ')[:160]}...")
