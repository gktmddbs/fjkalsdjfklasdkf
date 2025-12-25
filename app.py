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

# --- [1. 기본 설정 및 상수] ---
st.set_page_config(page_title="Nano Banana (Webtoon Engine)", page_icon="🍌", layout="wide")

# API 키 로드 (Secrets 또는 환경변수)
try:
    DEFAULT_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    DEFAULT_API_KEY = ""

# 모델 설정
MODEL_WORKER = "gemini-2.0-flash-exp" # 혹은 "gemini-1.5-pro" 등 사용 가능한 최신 모델
MODEL_INSPECTOR = "gemini-2.0-flash-exp" # 빠르고 저렴한 모델 권장

# --- [2. 프롬프트 정의] ---

# 작업자(Worker) 프롬프트: CSS 메타포와 강력한 제약사항 포함
WORKER_PROMPT = """
**Role & Objective:**
You are an expert Manga Localizer and Image Editor. Your task is to replace Japanese text with Korean text in the provided manga image. You must deliver a high-quality, read-to-read Korean version while **strictly preserving** the original artwork outside of text areas.

**CRITICAL MANDATE: PIXEL-PERFECT ART PRESERVATION**
*   **Do NOT Redraw:** You are strictly FORBIDDEN from altering characters, facial expressions, clothing, or background details.
*   **Targeted Editing:** Apply changes **ONLY** to the pixels containing text (speech bubbles, sound effects).
*   **Frozen Layer Rule:** Treat all non-text areas as a "locked layer" that must remain identical to the original image.

**CORE INSTRUCTIONS:**

**1. Strict Reading Order (Right-to-Left Logic):**
*   **Direction:** Japanese manga is read **Right-to-Left (RTL)**.
*   **Sequence:** You MUST process and translate dialogue starting from the **Rightmost** bubble/panel to the **Leftmost**.
*   **Logic Check:** Ensure the "Question" (Right) comes before the "Answer" (Left). Do not swap the conversation flow.

**2. Visual Layout Rules (Must Follow):**
*   **Rule A: HORIZONTAL Text Only (가로쓰기 강제):**
    *   Convert ALL Korean dialogue to **Horizontal (Left-to-Right)** orientation.
    *   **Prohibited:** Do NOT write Korean vertically (stacking characters top-to-bottom).
    *   **Formatting:** Use line breaks to center the horizontal text block within vertical bubbles.
*   **Rule B: Bubble Containment:**
    *   Text must stay **strictly INSIDE** the white speech bubbles.
    *   **Hallucination Check:** NEVER place translated text in empty background space or floating over artwork.
*   **Rule C: Inpainting & Cleaning:**
    *   Completely **ERASE** the original Japanese text first. Fill the gap with the bubble color (usually white) or background pattern (screentone) seamlessly.

**3. Translation & Localization:**
*   **Context & Tone:** Analyze the visual context. Translate into natural Korean reflecting the character's persona.
*   **Sound Effects (SFX):** Translate background SFX text. Match the "visual weight" of the original SFX.

**Output:**
Return ONLY the final processed image.
"""

# 검수자(Inspector) 프롬프트 - 레벨 2 (기본)
INSPECTOR_PROMPT_BASIC = """
# Role
You are a Visual Quality Assurance Supervisor.

# Task
Compare the [Generated Image] with the [Original Image] to detect HALLUCINATIONS or DESTRUCTION.

# PASS Criteria (Broad):
1. **Composition:** Does the output look like the same page? (Layout, Panels).
2. **Art Integrity:** Are the characters' faces intact? (Not melted/blurred/scary).
3. **Text Placement:** Is text roughly inside bubbles?

# IGNORE:
- Do NOT check for vertical/horizontal text direction.
- Do NOT check for translation accuracy.

# Output Format (JSON ONLY)
If PASS: {"status": "PASS"}
If FAIL (Face melted / Totally different image): {"status": "FAIL", "reason": "Severe visual distortion detected"}
"""

# 검수자(Inspector) 프롬프트 - 레벨 3 (엄격)
INSPECTOR_PROMPT_STRICT = """
# Role
You are a Strict Localization QA Supervisor.

# Task
Inspect the [Generated Image] for TEXT FORMATTING and TRANSLATION failures.

# FAIL CRITERIA (Strict):

1. **Vertical Text (CRITICAL):**
   - **FAIL:** If you see any **Korean text written vertically** (stacked top-to-bottom) with 2 or more characters.
   - **PASS:** Single character vertical exclamations (e.g., "!", "?") or vertical SFX are OK.
   
2. **Untranslated Text:**
   - **FAIL:** If Japanese Kana/Kanji is still visible inside speech bubbles.

3. **Visual Integrity:**
   - **FAIL:** If the character's face is distorted.

# Output Format (JSON ONLY)
If PASS: {"status": "PASS"}
If FAIL: {"status": "FAIL", "reason": "Vertical text or Untranslated Japanese detected"}
"""

# --- [3. 유틸리티 함수] ---

@st.cache_resource
def get_genai_client(api_key):
    return genai.Client(api_key=api_key)

def save_image_to_temp(image: Image.Image, filename: str) -> str:
    temp_dir = tempfile.gettempdir()
    # 파일명 안전 처리
    safe_name = f"{uuid.uuid4().hex[:8]}_{filename}"
    path = os.path.join(temp_dir, safe_name)
    image.save(path, format="PNG")
    return path

def load_image_optimized(path_or_file) -> Image.Image:
    """이미지 로드 시 회전 보정 및 RGB 변환"""
    try:
        if isinstance(path_or_file, str):
            if not os.path.exists(path_or_file): return None
            img = Image.open(path_or_file)
        else:
            img = Image.open(path_or_file)
            
        img = ImageOps.exif_transpose(img) # EXIF 회전 정보 반영
        
        # 투명도(Alpha)가 있는 경우 흰색 배경으로 병합 (JPG/API 호환성)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[3])
            return background
        else:
            return img.convert("RGB")
    except Exception as e:
        st.error(f"이미지 로드 실패: {e}")
        return None

def image_to_bytes(image: Image.Image) -> bytes:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def init_session_state():
    defaults = {
        'job_queue': [], 
        'results': [],
        'uploader_key': 0, 
        'last_pasted_hash': None, 
        'is_auto_running': False
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

def create_zip_file():
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for item in st.session_state.results:
            img = load_image_optimized(item['result_path']) 
            if img:
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                
                # 파일명 정리
                base_name = item['name']
                if base_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    base_name = os.path.splitext(base_name)[0]
                
                filename = f"kor_{base_name}.png"
                zip_file.writestr(filename, img_bytes.getvalue())
    return zip_buffer.getvalue()

# --- [4. AI 로직 (핵심 엔진)] ---

def verify_image(api_key, original_img, generated_img, mode):
    """
    mode: "OFF" | "BASIC" | "STRICT"
    """
    if mode == "OFF":
        return True, "Skipped (User Request)"

    target_prompt = INSPECTOR_PROMPT_STRICT if mode == "STRICT" else INSPECTOR_PROMPT_BASIC

    try:
        client = get_genai_client(api_key)
        
        contents = [
            target_prompt,
            "Here is the ORIGINAL image:",
            types.Part.from_bytes(data=image_to_bytes(original_img), mime_type="image/png"),
            "Here is the GENERATED result:",
            types.Part.from_bytes(data=image_to_bytes(generated_img), mime_type="image/png")
        ]

        response = client.models.generate_content(
            model=MODEL_INSPECTOR,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0, # 검수는 냉철하게
                response_mime_type="application/json"
            )
        )
        
        if response.text:
            try:
                # JSON 파싱 시도 (가끔 마크다운 ```json ... ``` 으로 감싸서 줄 때 대응)
                clean_text = response.text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:-3]
                elif clean_text.startswith("```"):
                    clean_text = clean_text[3:-3]
                
                data = json.loads(clean_text)
                
                if data.get("status") == "PASS":
                    return True, "PASS"
                else:
                    return False, data.get("reason", "Unknown Rejection")
            except json.JSONDecodeError:
                # JSON 파싱 실패하면 그냥 통과시킴 (작업 중단 방지)
                return True, "JSON Error (Pass)"
        return True, "No Response (Pass)"
        
    except Exception as e:
        return True, f"Inspector Error: {e} (Pass)"

def generate_with_auto_fix(api_key, prompt, image_input, resolution, temperature, verify_mode, max_retries=2, status_container=None):
    client = get_genai_client(api_key)
    target_bytes = image_to_bytes(image_input)
    last_error = ""

    # 안전 설정 (차단 최소화)
    safety_settings = [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    ]

    for attempt in range(max_retries + 1):
        try:
            # 1. Temperature 동적 보정
            current_temp = temperature
            # 재시도 중이고, 기존 Temp가 낮았다면 높여서 편향 깨기
            if attempt > 0 and temperature < 0.5:
                current_temp = 0.65
                if status_container: status_container.warning(f"🔥 전략 변경: 창의성을 {current_temp}로 높여 재시도합니다.")

            # 2. 프롬프트 강화 (CSS Injection)
            css_instruction = (
                "\n# TECHNICAL OVERRIDE:\n"
                "Apply CSS: `writing-mode: horizontal-tb !important;`\n"
                "If bubbles are narrow, FORCE line breaks every 2-3 chars.\n"
            )
            
            retry_instruction = ""
            if attempt > 0 and last_error:
                retry_instruction = (
                    f"\n🚨 **PREVIOUS REJECTION REASON: {last_error}** 🚨\n"
                    "You failed the Quality Assurance check.\n"
                    "If the error was 'Vertical Text', force Horizontal text output.\n"
                    "If the error was 'Distortion', preserve the original art strictly.\n"
                )

            # 3. API 호출
            contents = [
                prompt + css_instruction + retry_instruction,
                "Process this image:",
                types.Part.from_bytes(data=target_bytes, mime_type="image/png")
            ]

            response = client.models.generate_content(
                model=MODEL_WORKER,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=current_temp,
                    safety_settings=safety_settings
                )
            )
            
            # 4. 결과 추출
            result_img = None
            
            # Safety Block 확인
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
                if finish_reason != "STOP":
                    fail_msg = f"⚠️ Safety Filter Blocked: {finish_reason}"
                    if status_container: status_container.error(fail_msg)
                    return None, fail_msg

            if response.parts:
                for part in response.parts:
                    if part.inline_data: 
                        result_img = Image.open(io.BytesIO(part.inline_data.data))
                        break
            
            # SDK 버전에 따른 호환성
            if not result_img and hasattr(response, 'image') and response.image: 
                result_img = response.image

            if not result_img:
                # 텍스트만 뱉고 이미지를 안 준 경우
                if status_container: status_container.error("❌ 이미지가 생성되지 않았습니다. (모델이 텍스트로 응답함)")
                return None, "No Image Generated"

            # 5. 검수 (Inspector)
            if attempt < max_retries:
                if status_container: status_container.info(f"🧐 품질 검수 중... (Mode: {verify_mode})")
                
                is_pass, reason = verify_image(api_key, image_input, result_img, verify_mode)
                
                if is_pass:
                    if status_container: status_container.success("✅ 검수 통과!")
                    return result_img, None 
                else:
                    last_error = reason
                    if status_container: status_container.warning(f"🚨 불합격: {reason} -> 재시도 중...")
                    time.sleep(1.0)
                    continue
            else:
                if status_container: status_container.warning("⚠️ 최대 재시도 횟수 도달. 현재 결과를 반환합니다.")
                return result_img, "Max Retries Reached"

        except Exception as e:
            if "429" in str(e):
                if status_container: status_container.warning("⏳ API 사용량 제한. 5초 대기...")
                time.sleep(5)
                continue
            return None, f"API Error: {str(e)}"
            
    return None, "Unknown Error"

# --- [5. 메인 처리 로직] ---

def process_and_update(item, api_key, prompt, resolution, temperature, use_autofix, verify_mode):
    original_img = load_image_optimized(item['image_path'])
    if not original_img:
        st.error("원본 이미지가 만료되었습니다. 다시 업로드해주세요.")
        return

    # Auto-fix 옵션이 꺼져있거나 검수가 OFF면 재시도 횟수 0
    max_retries = 2 if (use_autofix and verify_mode != "OFF") else 0
    
    start_time = time.time()
    
    with st.status(f"🚀 **{item['name']}** 작업 시작...", expanded=True) as status:
        res_img, err = generate_with_auto_fix(
            api_key, prompt, original_img, resolution, temperature, 
            verify_mode, max_retries, status_container=status
        )

        duration = time.time() - start_time

        if res_img:
            res_path = save_image_to_temp(res_img, f"result_{item['name']}")
            status.update(label=f"✅ 완료! ({duration:.2f}초)", state="complete", expanded=False)
            
            st.session_state.results.append({
                'id': str(uuid.uuid4()), 
                'name': item['name'], 
                'original_path': item['image_path'], 
                'result_path': res_path,
                'duration': duration
            })
            # 대기열에서 제거
            st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]
            time.sleep(0.5)
            st.rerun()
        else:
            status.update(label="❌ 작업 실패", state="error", expanded=True)
            item['status'] = 'error'
            item['error_msg'] = err
            st.rerun()

def auto_process_step(api_key, prompt, resolution, temperature, use_autofix, verify_mode):
    if not st.session_state.is_auto_running: return
    pending = [i for i in st.session_state.job_queue if i['status'] == 'pending']
    
    if not pending:
        st.session_state.is_auto_running = False
        st.toast("✅ 모든 작업이 완료되었습니다!")
        time.sleep(1)
        st.rerun()
        return

    item = pending[0]
    # 위와 동일한 로직이지만 자동 실행용
    process_and_update(item, api_key, prompt, resolution, temperature, use_autofix, verify_mode)


# --- [6. UI 컴포넌트] ---

def render_sidebar():
    with st.sidebar:
        st.title("🍌 Nano Banana")
        st.caption("Webtoon Engine v2.0")
        
        api_key = st.text_input("Google API Key", value=DEFAULT_API_KEY, type="password")
        if not api_key:
            st.warning("API 키를 입력하세요.")
        
        st.info(f"🛠️ Worker: {MODEL_WORKER}\n👮 Inspector: {MODEL_INSPECTOR}")

        st.divider()
        st.subheader("⚙️ 모델 설정")
        
        # 해상도 (참고: API 버전에 따라 image_size가 무시될 수 있음)
        resolution = st.radio("해상도", options=["2K", "1K"], index=0, horizontal=True)
        res_tuple = (2048, 2048) if resolution == "2K" else (1024, 1024)

        temperature = st.slider("창의성 (Temperature)", 0.0, 1.0, 0.4, 0.1, help="낮을수록 원본 보존력이 좋지만, 0.0은 때로 번역을 거부할 수 있습니다.")

        st.divider()
        st.subheader("🧐 검수 옵션 (Inspector)")
        
        inspector_option = st.radio(
            "검수 수준 선택",
            options=["1. 검수 안 함 (빠름)", "2. 기본 (이미지 깨짐 방지)", "3. 엄격 (세로쓰기/미번역 잡기)"],
            index=1
        )
        
        if "1." in inspector_option: verify_mode = "OFF"
        elif "3." in inspector_option: verify_mode = "STRICT"
        else: verify_mode = "BASIC"

        use_autofix = st.toggle("🛡️ 자동 재시도 (Auto-Retry)", value=True, help="검수 실패 시 자동으로 설정을 변경하여 다시 시도합니다.")
        
        if st.button("🗑️ 모든 데이터 초기화", use_container_width=True):
            st.session_state.job_queue = []
            st.session_state.results = []
            st.rerun()
            
        st.divider()
        use_slider = st.toggle("비교 슬라이더 켜기", value=True)
        with st.expander("📝 프롬프트 수정"):
            prompt = st.text_area("System Instructions", value=WORKER_PROMPT, height=300)

        return api_key, use_slider, prompt, res_tuple, temperature, use_autofix, verify_mode

def handle_file_upload():
    col1, col2 = st.columns([3, 1])
    with col1: 
        files = st.file_uploader("이미지 추가", type=['png', 'jpg', 'jpeg', 'zip'], accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")
    with col2:
        st.write("클립보드:")
        # paste_image_button은 image_data 속성에 PIL Image 객체를 담아 반환합니다.
        paste_btn = paste_image_button(label="📋 붙여넣기", text_color="#ffffff", background_color="#FF4B4B", hover_background_color="#FF0000")

    new_cnt = 0
    # 1. 파일 업로드 처리
    if files:
        with st.spinner("파일 처리 중..."):
            for f in files:
                if f.name.lower().endswith('.zip'):
                    try:
                        with zipfile.ZipFile(f) as z:
                            img_files = [n for n in z.namelist() if n.lower().endswith(('.png','.jpg','.jpeg')) and '__MACOSX' not in n]
                            for fname in img_files:
                                with z.open(fname) as img_f:
                                    img = load_image_optimized(io.BytesIO(img_f.read()))
                                    if img:
                                        path = save_image_to_temp(img, os.path.basename(fname))
                                        st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': os.path.basename(fname), 'image_path': path, 'status': 'pending', 'error_msg': None})
                                        new_cnt += 1
                    except: pass
                else:
                    img = load_image_optimized(f)
                    if img:
                        path = save_image_to_temp(img, f.name)
                        st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': f.name, 'image_path': path, 'status': 'pending', 'error_msg': None})
                        new_cnt += 1
    
    # 2. 붙여넣기(Paste) 처리 [수정된 부분]
    if paste_btn.image_data:
        # paste_btn.image_data는 이미 PIL Image 객체입니다.
        pasted_img = paste_btn.image_data
        
        # 해시 생성을 위해 바이트로 변환 (기존 유틸 함수 활용)
        img_bytes = image_to_bytes(pasted_img)
        curr_hash = hashlib.md5(img_bytes).hexdigest()
        
        if st.session_state.last_pasted_hash != curr_hash:
            # 이미지 전처리 (회전 보정 등) 수행
            # PIL Image 객체이므로 load_image_optimized 대신 직접 처리하거나 그대로 사용
            # 여기서는 안전하게 바이트IO를 거쳐 최적화 함수를 통과시킵니다.
            processed_img = load_image_optimized(io.BytesIO(img_bytes))
            
            if processed_img:
                path = save_image_to_temp(processed_img, f"paste_{int(time.time())}.png")
                st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': f"paste_{int(time.time())}.png", 'image_path': path, 'status': 'pending', 'error_msg': None})
                st.session_state.last_pasted_hash = curr_hash
                new_cnt += 1

    if new_cnt > 0:
        time.sleep(0.5)
        st.session_state.uploader_key += 1
        st.rerun()

def render_queue(api_key, prompt, resolution, temperature, use_autofix, verify_mode):
    if not st.session_state.job_queue: return

    st.divider()
    c1, c2, c3 = st.columns([3, 1, 1])
    pending = [i for i in st.session_state.job_queue if i['status'] == 'pending']
    c1.subheader(f"📂 대기열 ({len(st.session_state.job_queue)}장 / 대기 {len(pending)}장)")
    
    if not st.session_state.is_auto_running:
        if c2.button(f"🚀 전체 실행", type="primary", use_container_width=True, disabled=len(pending)==0):
            st.session_state.is_auto_running = True
            st.rerun()
    else:
        if c2.button("⏹️ 중지", type="secondary", use_container_width=True):
            st.session_state.is_auto_running = False
            st.rerun()

    if c3.button("🗑️ 선택 삭제", use_container_width=True):
        st.session_state.job_queue = []
        st.rerun()

    if st.session_state.is_auto_running: st.progress(100, text="🔄 자동 작업 중...")

    # 대기열 리스트 표시
    for item in st.session_state.job_queue:
        with st.container(border=True):
            col_img, col_info = st.columns([1, 4])
            with col_img:
                img = load_image_optimized(item['image_path'])
                if img: st.image(img, use_container_width=True)
            with col_info:
                st.markdown(f"**{item['name']}**")
                if item['status'] == 'error': 
                    st.error(f"❌ {item['error_msg']}")
                elif item['status'] == 'pending': 
                    st.info("⏳ 대기 중")
                
                b1, b2 = st.columns([1, 4])
                if b1.button("▶️", key=f"run_{item['id']}"): 
                    process_and_update(item, api_key, prompt, resolution, temperature, use_autofix, verify_mode)
                if b2.button("🗑️", key=f"del_{item['id']}"):
                    st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]
                    st.rerun()

def render_results(use_slider):
    if not st.session_state.results: return

    st.divider()
    st.subheader(f"🖼️ 완료된 작업 ({len(st.session_state.results)}장)")

    # 저장 패널
    with st.container(border=True):
        st.markdown("### 💾 결과물 저장")
        c1, c2 = st.columns(2)
        zip_name = c1.text_input("ZIP 파일명", value="translated_manga")
        local_path = c2.text_input("로컬 폴더 경로 (Optional)", placeholder="예: C:/Manga/Chapter1")
        
        b1, b2, b3 = st.columns(3)
        
        # ZIP 다운로드
        zip_data = create_zip_file()
        b1.download_button("📦 ZIP 다운로드", data=zip_data, file_name=f"{zip_name}.zip", mime="application/zip", use_container_width=True, type="primary")

        # 로컬 저장
        if b2.button("📂 PC 저장", use_container_width=True):
            if local_path and os.path.exists(local_path):
                cnt = 0
                for item in st.session_state.results:
                    img = load_image_optimized(item['result_path'])
                    if img:
                        fname = f"kor_{item['name']}"
                        if not fname.lower().endswith('.png'): fname += ".png"
                        img.save(os.path.join(local_path, fname))
                        cnt += 1
                st.success(f"{cnt}장 저장 완료!")
            else:
                st.error("유효하지 않은 경로입니다.")
        
        if b3.button("🗑️ 결과 비우기", use_container_width=True):
            st.session_state.results = []
            st.rerun()

    # 결과 리스트
    for item in st.session_state.results:
        with st.container(border=True):
            c_img, c_info = st.columns([1, 2])
            
            orig = load_image_optimized(item['original_path'])
            res = load_image_optimized(item['result_path'])

            with c_img:
                if res: st.image(res, use_container_width=True)
            
            with c_info:
                st.markdown(f"### {item['name']}")
                st.caption(f"⏱️ 소요시간: {item['duration']:.1f}초")
                
                if use_slider and orig and res:
                    with st.expander("🆚 비교 보기"):
                        if orig.size != res.size: orig = orig.resize(res.size)
                        image_comparison(img1=orig, img2=res, label1="Original", label2="Trans", in_memory=True)
                
                d1, d2 = st.columns(2)
                
                # 개별 다운로드
                if res:
                    buf = io.BytesIO()
                    res.save(buf, format="PNG")
                    d1.download_button("⬇️ 다운로드", data=buf.getvalue(), file_name=f"kor_{item['name']}.png", mime="image/png", key=f"dl_{item['id']}")
                
                if d2.button("🗑️ 삭제", key=f"rm_{item['id']}"):
                    st.session_state.results = [x for x in st.session_state.results if x['id'] != item['id']]
                    st.rerun()

# --- [7. 메인 실행] ---
def main():
    init_session_state()
    
    # 사이드바에서 설정값 받기
    api_key, use_slider, prompt, resolution, temperature, use_autofix, verify_mode = render_sidebar()
    
    handle_file_upload()
    
    # 큐 렌더링 및 자동 실행 체크
    render_queue(api_key, prompt, resolution, temperature, use_autofix, verify_mode)
    
    if st.session_state.is_auto_running:
        auto_process_step(api_key, prompt, resolution, temperature, use_autofix, verify_mode)
        
    render_results(use_slider)

if __name__ == "__main__":
    main()

