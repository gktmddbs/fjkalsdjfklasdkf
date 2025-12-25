import streamlit as st
from google import genai
from google.genai import types
from PIL import Image, ImageOps
import io
import os
import time
import uuid
import hashlib
import zipfile
import tempfile
import json
from streamlit_paste_button import paste_image_button
from streamlit_image_comparison import image_comparison

# [추가] RAR 지원을 위한 라이브러리 (pip install rarfile 필요)
try:
    import rarfile
    # Windows: WinRAR 경로 자동 탐색
    if os.name == 'nt':
        rar_paths = [r"C:\Program Files\WinRAR\UnRAR.exe", r"C:\Program Files (x86)\WinRAR\UnRAR.exe"]
        for p in rar_paths:
            if os.path.exists(p):
                rarfile.UNRAR_TOOL = p
                break
except ImportError:
    rarfile = None

# --- [1. 기본 설정 및 상수] ---
st.set_page_config(page_title="Nano Banana v3.1 (Gemini 3 Edition)", page_icon="🍌", layout="wide")

# API 키 로드
try:
    DEFAULT_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    DEFAULT_API_KEY = ""

# -------------------------------------------------------------------------
# [모델 설정 변경] 요청하신 Gemini 3 Pro 및 Flash 적용
# -------------------------------------------------------------------------
# 작업자(Worker): 고성능 이미지 편집 및 추론용
MODEL_WORKER = "gemini-3-pro-image-preview" 
# 검수자(Inspector): 빠른 비전 인식 및 검수용
MODEL_INSPECTOR = "gemini-3-flash-preview"
# -------------------------------------------------------------------------

# --- [2. 프롬프트 정의 (v3.0 Diamond Engine)] ---

# 작업자(Worker) 프롬프트
WORKER_PROMPT = """
**SYSTEM IDENTITY:**
You are a Veteran Manga Localizer & Typesetter with 20+ years of experience.
Your goal is to translate Japanese manga to Korean, performing simultaneous **Inpainting (Cleaning)** and **Typesetting (Rendering)**.

**CORE PROCESS (Chain of Thought):**
Before rendering, you must internally process these steps:
1.  **Analyze:** Identify Reading Order (RTL), Bubble Shapes (Oval, Spiky, Cloud), and Character Emotions.
2.  **Translate:** Translate text into context-aware Korean.
3.  **Plan Layout:** Apply the "Diamond Typesetting" logic for vertical bubbles.
4.  **Render:** Erase original text seamlessly and render the new Korean text.

**CRITICAL RULES:**

**1. "DIAMOND" TYPESETTING ALGORITHM (For Vertical Bubbles):**
*   **Problem:** Korean text is horizontal, but Japanese bubbles are vertical.
*   **Solution:** You MUST arrange line lengths in a **`Short - Long - Short`** pattern to mimic the bubble shape.
*   **Constraint:** Insert line breaks (`\\n`) aggressively. Do NOT write long single lines.
    *   *BAD:* [용서하지않겠다] (1 line)
    *   *GOOD:*
        [용서하지]
        [않겠다!] (Diamond shape)

**2. EMOTION-TO-FONT MAPPING:**
Observe the bubble shape and emotion to select the font style:
*   **Standard/Oval Bubble** -> Use **Clean Gothic (Sans-serif)**. (Legibility is key)
*   **Shouting/Spiky Bubble** -> Use **Extra Bold / Impact** style. (Thick, heavy strokes)
*   **Thought/Cloud Bubble** -> Use **Handwritten / Thin** style. (Soft, dreamy look)
*   **Narration/Square Box** -> Use **Serif (Myeongjo)** style. (Serious tone)

**3. VISUAL SFX TRANSLATION:**
*   Translate Sound Effects (SFX) by matching their **Visual Weight**.
*   If original is Rough/Brush -> Korean SFX must be Rough/Brush.
*   If original is Neon/Glow -> Korean SFX must have Glow.

**4. NEGATIVE CONSTRAINTS:**
*   **NO Vertical Text Columns:** Korean text must be horizontal.
*   **NO Text Overflow:** Text must stay strictly inside the white bubble area.
*   **NO Hallucinations:** Do not generate text in empty background space.
*   **Pixel-Perfect Art:** Do NOT redraw characters or backgrounds. Only touch the text areas.

**Output:**
Return ONLY the final processed image.
"""

# 검수자(Inspector) 프롬프트 - 기본
INSPECTOR_PROMPT_BASIC = """
# Role: Visual QA Supervisor
# Task: Detect HALLUCINATIONS or DESTRUCTION.
# PASS Criteria:
1. Composition matches original.
2. Characters' faces are intact (not melted).
3. Text is roughly inside bubbles.
# Output: JSON {"status": "PASS" or "FAIL", "reason": "..."}
"""

# 검수자(Inspector) 프롬프트 - 엄격
INSPECTOR_PROMPT_STRICT = """
# Role: Strict QA Supervisor
# Task: Detect VERTICAL TEXT & UNTRANSLATED TEXT.
# FAIL Criteria:
1. **Vertical Text:** Any Korean text stacked vertically (2+ chars).
2. **Untranslated:** Japanese Kana/Kanji remaining.
3. **Distortion:** Character face distortion.
# Output: JSON {"status": "PASS" or "FAIL", "reason": "..."}
"""

# --- [3. 유틸리티 함수] ---

@st.cache_resource
def get_genai_client(api_key):
    return genai.Client(api_key=api_key)

def image_to_bytes(image: Image.Image) -> bytes:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def save_image_to_temp(image: Image.Image, filename: str) -> str:
    temp_dir = tempfile.gettempdir()
    safe_name = f"{uuid.uuid4().hex[:8]}_{filename}"
    path = os.path.join(temp_dir, safe_name)
    image.save(path, format="PNG")
    return path

def load_image_optimized(path_or_file) -> Image.Image:
    try:
        if isinstance(path_or_file, str):
            if not os.path.exists(path_or_file): return None
            img = Image.open(path_or_file)
        else:
            img = Image.open(path_or_file)
            
        img = ImageOps.exif_transpose(img)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == 'P': img = img.convert('RGBA')
            background.paste(img, mask=img.split()[3])
            return background
        return img.convert("RGB")
    except Exception:
        return None

def convert_rar_to_zip_memory(rar_bytes):
    """RAR 바이너리를 메모리 상에서 ZIP 바이너리로 변환"""
    if rarfile is None: raise ImportError("rarfile 라이브러리 없음")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".rar") as tmp_rar:
        tmp_rar.write(rar_bytes)
        tmp_rar_path = tmp_rar.name

    zip_buffer = io.BytesIO()
    try:
        with rarfile.RarFile(tmp_rar_path) as rf:
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for filename in rf.namelist():
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_data = rf.read(filename)
                        zf.writestr(filename, file_data)
    finally:
        if os.path.exists(tmp_rar_path): os.remove(tmp_rar_path)
    
    zip_buffer.seek(0)
    return zip_buffer

def create_zip_file():
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for item in st.session_state.results:
            img = load_image_optimized(item['result_path']) 
            if img:
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                base = os.path.splitext(item['name'])[0]
                zip_file.writestr(f"kor_{base}.png", img_bytes.getvalue())
    return zip_buffer.getvalue()

def init_session_state():
    if 'job_queue' not in st.session_state: st.session_state.job_queue = []
    if 'results' not in st.session_state: st.session_state.results = []
    if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0
    if 'last_pasted_hash' not in st.session_state: st.session_state.last_pasted_hash = None
    if 'is_auto_running' not in st.session_state: st.session_state.is_auto_running = False

# --- [4. AI 로직] ---

def verify_image(api_key, original_img, generated_img, mode):
    if mode == "OFF": return True, "Skipped"
    target_prompt = INSPECTOR_PROMPT_STRICT if mode == "STRICT" else INSPECTOR_PROMPT_BASIC
    
    try:
        client = get_genai_client(api_key)
        response = client.models.generate_content(
            model=MODEL_INSPECTOR,
            contents=[
                target_prompt,
                "ORIGINAL:", types.Part.from_bytes(data=image_to_bytes(original_img), mime_type="image/png"),
                "GENERATED:", types.Part.from_bytes(data=image_to_bytes(generated_img), mime_type="image/png")
            ],
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
        )
        if response.text:
            text = response.text.strip().replace("```json", "").replace("```", "")
            data = json.loads(text)
            if data.get("status") == "PASS": return True, "PASS"
            return False, data.get("reason", "Rejected")
    except:
        return True, "Error (Pass)"
    return True, "No Response"

def generate_with_auto_fix(api_key, prompt, image_input, resolution, temperature, verify_mode, max_retries=2, status_container=None):
    client = get_genai_client(api_key)
    target_bytes = image_to_bytes(image_input)
    last_error = ""

    safety = [types.SafetySetting(category=cat, threshold="BLOCK_NONE") for cat in 
              ["HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_DANGEROUS_CONTENT", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_HARASSMENT"]]

    for attempt in range(max_retries + 1):
        try:
            curr_temp = temperature
            if attempt > 0 and temperature < 0.5: curr_temp = 0.65

            css_inject = "\n# LAYOUT OVERRIDE: Apply 'Diamond Shape' text wrapping. Force horizontal writing."
            retry_msg = f"\n🚨 PREVIOUS ERROR: {last_error}. FIX IT." if last_error else ""
            
            response = client.models.generate_content(
                model=MODEL_WORKER,
                contents=[prompt + css_inject + retry_msg, types.Part.from_bytes(data=target_bytes, mime_type="image/png")],
                config=types.GenerateContentConfig(temperature=curr_temp, safety_settings=safety)
            )

            result_img = None
            if response.candidates and response.candidates[0].finish_reason != "STOP":
                return None, f"Safety Block: {response.candidates[0].finish_reason}"
            
            if response.parts:
                for part in response.parts:
                    if part.inline_data:
                        result_img = Image.open(io.BytesIO(part.inline_data.data))
                        break
            if not result_img and hasattr(response, 'image'): result_img = response.image

            if not result_img: return None, "No Image Generated"

            if attempt < max_retries:
                if status_container: status_container.info(f"🧐 검수 중... (Mode: {verify_mode})")
                is_pass, reason = verify_image(api_key, image_input, result_img, verify_mode)
                if is_pass: return result_img, None
                last_error = reason
                if status_container: status_container.warning(f"🚨 재시도: {reason}")
                time.sleep(1)
                continue
            
            return result_img, "Max Retries"

        except Exception as e:
            if "429" in str(e): time.sleep(5); continue
            return None, str(e)
    return None, "Unknown"

# --- [5. 메인 처리 로직] ---

def process_and_update(item, api_key, prompt, resolution, temperature, use_autofix, verify_mode):
    img = load_image_optimized(item['image_path'])
    if not img: return

    max_retries = 2 if (use_autofix and verify_mode != "OFF") else 0
    start = time.time()
    
    with st.status(f"🚀 **{item['name']}** 처리 중...", expanded=True) as status:
        res, err = generate_with_auto_fix(api_key, prompt, img, resolution, temperature, verify_mode, max_retries, status)
        
        if res:
            path = save_image_to_temp(res, f"res_{item['name']}")
            status.update(label="✅ 완료!", state="complete", expanded=False)
            st.session_state.results.append({
                'id': str(uuid.uuid4()), 'name': item['name'], 
                'original_path': item['image_path'], 'result_path': path, 
                'duration': time.time() - start
            })
            st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]
            time.sleep(0.5)
            st.rerun()
        else:
            status.update(label="❌ 실패", state="error")
            item['status'] = 'error'; item['error_msg'] = err
            st.rerun()

def auto_process(api_key, prompt, resolution, temperature, use_autofix, verify_mode):
    if not st.session_state.is_auto_running: return
    pending = [i for i in st.session_state.job_queue if i['status'] == 'pending']
    if not pending:
        st.session_state.is_auto_running = False
        st.toast("작업 완료!")
        time.sleep(1); st.rerun()
        return
    process_and_update(pending[0], api_key, prompt, resolution, temperature, use_autofix, verify_mode)

# --- [6. UI 로직] ---

def handle_file_upload():
    c1, c2 = st.columns([3, 1])
    with c1: 
        files = st.file_uploader("이미지/ZIP/RAR 추가", type=['png','jpg','jpeg','zip','rar'], accept_multiple_files=True, key=f"up_{st.session_state.uploader_key}")
    with c2: 
        st.write("클립보드")
        paste = paste_image_button("📋 붙여넣기", text_color="white", background_color="#FF4B4B")

    new_cnt = 0
    if files:
        with st.spinner("파일 분석 중..."):
            for f in files:
                ext = f.name.split('.')[-1].lower()
                target_zip = None
                
                if ext == 'rar':
                    try: target_zip = convert_rar_to_zip_memory(f.read())
                    except Exception as e: st.error(f"RAR 오류: {e}")
                elif ext == 'zip': target_zip = f
                
                if target_zip:
                    try:
                        with zipfile.ZipFile(target_zip) as z:
                            for n in z.namelist():
                                if n.lower().endswith(('.png','.jpg','.jpeg')) and '__macosx' not in n.lower():
                                    with z.open(n) as zf:
                                        img = load_image_optimized(io.BytesIO(zf.read()))
                                        if img:
                                            safe_name = f"{os.path.splitext(f.name)[0]}_{os.path.basename(n)}"
                                            path = save_image_to_temp(img, safe_name)
                                            st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': os.path.basename(path), 'image_path': path, 'status': 'pending', 'error_msg': None})
                                            new_cnt += 1
                    except Exception as e: st.error(f"압축 오류: {e}")
                elif ext in ['png','jpg','jpeg']:
                    img = load_image_optimized(f)
                    if img:
                        path = save_image_to_temp(img, f.name)
                        st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': f.name, 'image_path': path, 'status': 'pending', 'error_msg': None})
                        new_cnt += 1
    
    if paste.image_data:
        pasted_img = paste.image_data
        img_bytes = image_to_bytes(pasted_img)
        h = hashlib.md5(img_bytes).hexdigest()
        
        if st.session_state.last_pasted_hash != h:
            img = load_image_optimized(io.BytesIO(img_bytes))
            path = save_image_to_temp(img, f"paste_{int(time.time())}.png")
            st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': os.path.basename(path), 'image_path': path, 'status': 'pending', 'error_msg': None})
            st.session_state.last_pasted_hash = h
            new_cnt += 1

    if new_cnt > 0: time.sleep(0.5); st.session_state.uploader_key += 1; st.rerun()

def main():
    init_session_state()
    
    with st.sidebar:
        st.title("🍌 Nano Banana v3.1")
        st.caption("Gemini 3 Pro + Flash Edition")
        api_key = st.text_input("API Key", value=DEFAULT_API_KEY, type="password")
        
        st.divider()
        res = st.radio("Resolution", ["2K (권장)", "1K"], 0)
        res_val = (2048, 2048) if "2K" in res else (1024, 1024)
        temp = st.slider("Temperature", 0.0, 1.0, 0.3)
        
        st.divider()
        insp = st.radio("검수 모드", ["1. OFF", "2. BASIC", "3. STRICT"], 1)
        v_mode = "OFF" if "1." in insp else ("STRICT" if "3." in insp else "BASIC")
        autofix = st.toggle("Auto-Fix", True)
        
        if st.button("Clear All"): st.session_state.job_queue = []; st.session_state.results = []; st.rerun()
        
        with st.expander("Prompt"):
            prompt = st.text_area("Worker Prompt", WORKER_PROMPT, height=300)

    st.title("Nano Banana v3.1")
    st.markdown("**Gemini 3 Pro Diamond Engine**")
    
    handle_file_upload()
    
    if st.session_state.job_queue:
        st.divider()
        c1, c2, c3 = st.columns([3, 1, 1])
        cnt = len([i for i in st.session_state.job_queue if i['status']=='pending'])
        c1.subheader(f"대기열 ({cnt}장)")
        if not st.session_state.is_auto_running:
            if c2.button("🚀 전체 실행", type="primary", disabled=cnt==0, use_container_width=True):
                st.session_state.is_auto_running = True; st.rerun()
        else:
            if c2.button("⏹️ 중지", use_container_width=True):
                st.session_state.is_auto_running = False; st.rerun()
        if c3.button("🗑️ 삭제", use_container_width=True): st.session_state.job_queue = []; st.rerun()
        
        if st.session_state.is_auto_running: st.progress(100, "작업 중...")

        for item in st.session_state.job_queue:
            with st.container(border=True):
                c_img, c_info = st.columns([1, 5])
                with c_img: 
                    img = load_image_optimized(item['image_path'])
                    if img: st.image(img)
                with c_info:
                    st.write(f"**{item['name']}**")
                    if item['status'] == 'error': st.error(item['error_msg'])
                    elif item['status'] == 'pending': st.info("대기")
                    
                    b1, b2 = st.columns([1, 6])
                    if b1.button("▶", key=f"r_{item['id']}"): process_and_update(item, api_key, prompt, res_val, temp, autofix, v_mode)
                    if b2.button("🗑️", key=f"d_{item['id']}"): 
                        st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]; st.rerun()

    if st.session_state.is_auto_running:
        auto_process(api_key, prompt, res_val, temp, autofix, v_mode)

    if st.session_state.results:
        st.divider()
        st.subheader("결과물")
        
        with st.container(border=True):
            st.write("내보내기")
            cc1, cc2 = st.columns(2)
            zip_d = create_zip_file()
            cc1.download_button("📦 ZIP 다운로드", zip_d, "manga_kr.zip", "application/zip", use_container_width=True, type="primary")
            if cc2.button("🗑️ 결과 비우기", use_container_width=True): st.session_state.results = []; st.rerun()

        for item in st.session_state.results:
            with st.container(border=True):
                c1, c2 = st.columns([1, 2])
                orig = load_image_optimized(item['original_path'])
                res = load_image_optimized(item['result_path'])
                
                with c1: st.image(res)
                with c2:
                    st.write(f"**{item['name']}**")
                    with st.expander("비교"):
                        if orig and res:
                             if orig.size != res.size: orig = orig.resize(res.size)
                             image_comparison(orig, res, "Original", "Translated", in_memory=True)
                    
                    buf = io.BytesIO(); res.save(buf, format='PNG')
                    st.download_button("⬇️ 다운", buf.getvalue(), f"kor_{item['name']}", "image/png")

if __name__ == "__main__":
    main()

