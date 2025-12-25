import os
import time
import json
import pandas as pd
import torch
import google.generativeai as genai
import gradio as gr
from PIL import Image
from datetime import datetime
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# ==========================================
# 1. КОНФИГУРАЦИЯ И ПУТИ
# ==========================================
GEMINI_API_KEY = "YOUR_KEY"
DB_PATH = "./recipe_db"
CSV_PATH = "data/all_recepies_inter.csv"
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel('gemini-2.5-flash')

# ==========================================
# 2. ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ
# ==========================================
print("Загрузка моделей (может занять время)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Эмбеддинги
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Florence-2 (Vision)
vlm_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).to(device).eval()
vlm_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

# Векторная база (RAG)
def get_db():
    if os.path.exists(DB_PATH):
        return Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    
    print("Создание базы данных из CSV...")
    df = pd.read_csv(CSV_PATH, sep='\t')
    docs = []
    for _, row in df.iterrows():
        content = f"{row['Name']} {row['composition']}"
        meta = {"name": str(row['Name']), "composition": str(row['composition']), "instructions": str(row['Инструкции'])}
        docs.append(Document(page_content=content, metadata=meta))
    return Chroma.from_documents(docs, embedding_model, persist_directory=DB_PATH)

vector_db = get_db()

# ==========================================
# 3. МЕТРИКИ И ОТЧЕТЫ
# ==========================================
def calculate_relevance(detected, matched_comp):
    """Считает коэффициент пересечения слов (0.0 - 1.0)"""
    d_set = set(detected.lower().replace(',', '').split())
    m_set = set(matched_comp.lower().replace(',', '').split())
    if not d_set: return 0.0
    return len(d_set.intersection(m_set)) / len(d_set)

def save_experiment_report(data):
    """Сохраняет JSON отчет в папку reports"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = REPORT_DIR / f"report_{timestamp}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return file_path

# ==========================================
# 4. ЛОГИКА ПАЙПЛАЙНА
# ==========================================
def process_request(image, weight, height, age, gender, activity, goal):
    start_total = time.time()
    
    try:
        # ШАГ 1: Vision (Florence-2)
        start_step = time.time()
        inputs = vlm_processor(text="<DETAILED_CAPTION>", images=image, return_tensors="pt").to(device)
        generated_ids = vlm_model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=512)
        generated_text = vlm_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = vlm_processor.post_process_generation(generated_text, task="<DETAILED_CAPTION>", image_size=(image.width, image.height))
        detected = parsed["<DETAILED_CAPTION>"]
        v_time = time.time() - start_step

        # ШАГ 2: RAG (Поиск)
        start_step = time.time()
        docs = vector_db.similarity_search(detected, k=1)
        r_time = time.time() - start_step
        source = docs[0].metadata if docs else None

        # ШАГ 3: LLM (Gemini)
        start_step = time.time()
        prompt = f"""
        Ты эксперт-диетолог. Пользователь: {gender}, {weight}кг, {height}см, цель: {goal}.
        На фото обнаружено: {detected}.

Базовый рецепт из CSV: {source['name']} ({source['composition']}). Инструкция: {source['instructions']}.
        
        Выполни:
        1. Расчет КБЖУ.
        2. Адаптацию рецепта под продукты на фото.
        3. Расчет числа дней питания этими продуктами.
        """
        response = llm_model.generate_content(prompt)
        l_time = time.time() - start_step
        
        total_time = time.time() - start_total

        # СБОР МЕТРИК
        relevance = calculate_relevance(detected, source['composition']) if source else 0
        report = {
            "image_analysis": detected,
            "matched_recipe": source['name'] if source else "None",
            "timings": {"vision": v_time, "rag": r_time, "llm": l_time, "total": total_time},
            "metrics": {"relevance_score": relevance},
            "user_data": {"w": weight, "h": height, "goal": goal}
        }
        
        report_file = save_experiment_report(report)
        
        final_output = f"{response.text}\n\n---\n**Метрики системы:**\n"
        final_output += f"- Время: {total_time:.2f}с\n- Релевантность: {relevance:.2%}\n- Отчет: `{report_file}`"
        
        return final_output

    except Exception as e:
        return f"Ошибка пайплайна: {str(e)}"

# ==========================================
# 5. ИНТЕРФЕЙС
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Smart Nutritionist System v4.0 (VLM + RAG + Metrics)")
    
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="Загрузите фото продуктов")
            with gr.Row():
                w = gr.Number(label="Вес (кг)", value=70)
                h = gr.Number(label="Рост (см)", value=175)
                a = gr.Number(label="Возраст", value=25)
            goal = gr.Dropdown(["Сброс веса", "Поддержание", "Набор массы"], label="Цель")
            btn = gr.Button("Запустить анализ и сохранить отчет", variant="primary")
        
        with gr.Column():
            output = gr.Markdown()

    btn.click(process_request, [img, w, h, a, gr.State("Женский"), gr.State("Средняя"), goal], output)


demo.launch()