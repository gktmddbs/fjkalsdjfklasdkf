import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io
import os
import time
import uuid
import hashlib
import zipfile
import pickle
from streamlit_paste_button import paste_image_button
from streamlit_image_comparison import image_comparison

# --- [1. 기본 설정] ---
st.set_page_config(page_title="Nano Banana (Auto-Fix)", page_icon="🍌", layout="wide")

try:
    DEFAULT_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    DEFAULT_API_KEY = ""

# 모델 설정
MODEL_WORKER = "gemini-3-pro-image-preview"  # 작업자 (고화질)
MODEL_INSPECTOR = "gemini-3-flash-preview"     # 감독관 (빠름/검수용)

DEFAULT_EX_IN_PATH = "example_in.png"
DEFAULT_EX_OUT_PATH = "example_out.png"
MEMORY_FILE = "banana_memory.pkl"

# 작업자 프롬프트
DEFAULT_PROMPT = """
# Role
You are an expert Manga Typesetter & Translator. Your goal is to produce a "Production-Ready" localized image.

# Task
Translate the text in the image into [Korean] and render it directly onto the original image.

# 1. Visual Constraints [CRITICAL]
- **[STRICT] Orientation:** All text MUST be Horizontal (Left-to-Right). NEVER use vertical text.
- **Inpainting:** Completely erase the original text and reconstruct the background/artwork behind it seamlessly.
- **Line Art:** DO NOT damage, blur, or alter the artist's original pen lines.
- **Resolution:** Output in high-resolution (4K).

# 2. Typography & Formatting
- **Speech Bubbles:** Center the text. Ensure margins so text does not touch the bubble borders.
- **Sound Effects (SFX):** If translating SFX, use a font style that matches the original impact (Bold/Rough).
- **Font Style:**
  - Dialogue: Readable Sans-serif (Gothic style).
  - Monologue/Narration: Serif (Myeongjo style).

# 3. Translation Accuracy
- Context-aware translation based on facial expressions and scene atmosphere.
- Natural Korean spacing and grammar.

# Output
Return ONLY the processed image file. No explanations.
"""

# ✅ [NEW] 감독관 프롬프트
INSPECTOR_PROMPT = """
# Role
You are a QA Supervisor for Korean Manga Localization.

# Task
Compare the [Original Image] and the [Translated Result] and inspect for CRITICAL FAILURES.

# Checklist (Fail Conditions)
1. **Vertical Text:** Is there any Korean text written vertically (Top-to-Bottom)? -> If YES, FAIL.
2. **Text Overflow:** Is text touching the speech bubble borders or cropped? -> If YES, FAIL.
3. **Hallucination/Blur:** Is the image blurry, or are faces distorted? -> If YES, FAIL.
4. **Untranslated:** Is there any original Japanese/English text remaining? -> If YES, FAIL.
5. **Wrong Language:** Is the output text NOT Korean? -> If YES, FAIL.

# Output Protocol
- If NO errors found: Reply "PASS"
- If ANY error found: Reply "FAIL: [Brief Reason]" (e.g., "FAIL: Vertical text detected")
"""

# --- [2. 유틸리티] ---
def save_session_to_disk():
    try:
        state_data = {'job_queue': st.session_state.job_queue, 'results': st.session_state.results}
        with open(MEMORY_FILE, 'wb') as f: pickle.dump(state_data, f)
    except: pass

def load_session_from_disk():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'rb') as f:
                data = pickle.load(f)
                return data.get('job_queue', []), data.get('results', [])
        except: return [], []
    return [], []

def init_session_state():
    saved_queue, saved_results = load_session_from_disk()
    defaults = {
        'job_queue': saved_queue, 'results': saved_results,
        'uploader_key': 0, 'last_pasted_hash': None, 'is_auto_running': False
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

def clear_all_data():
    st.session_state.job_queue = []
    st.session_state.results = []
    if os.path.exists(MEMORY_FILE): os.remove(MEMORY_FILE)
    st.rerun()

def get_image_hash(image: Image.Image) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return hashlib.md5(img_byte_arr.getvalue()).hexdigest()

def image_to_bytes(image: Image.Image) -> bytes:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def create_zip_file():
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for item in st.session_state.results:
            img_bytes = io.BytesIO()
            item['result'].save(img_bytes, format='PNG')
            filename = f"kor_{item['name']}"
            if not filename.lower().endswith('.png'): filename = os.path.splitext(filename)[0] + ".png"
            zip_file.writestr(filename, img_bytes.getvalue())
    return zip_buffer.getvalue()

def save_to_local_folder(folder_name):
    if not folder_name: return
    try:
        os.makedirs(folder_name, exist_ok=True)
        count = 0
        for item in st.session_state.results:
            safe_name = f"kor_{item['name']}"
            if not safe_name.lower().endswith('.png'): safe_name = os.path.splitext(safe_name)[0] + ".png"
            item['result'].save(os.path.join(folder_name, safe_name), format="PNG")
            count += 1
        st.success(f"✅ 저장 완료: {count}장")
    except Exception as e: st.error(f"저장 실패: {e}")

@st.dialog("📷 이미지 전체 화면", width="large")
def show_full_image(image, caption):
    st.image(image, caption=caption, use_container_width=True)

# --- [3. AI 로직 (생성 + 검수)] ---

def verify_image(api_key, original_img, generated_img):
    """감독관(Flash)이 결과물을 검사하는 함수"""
    try:
        client = genai.Client(api_key=api_key)
        
        # 원본과 결과물을 비교하게 함
        contents = [
            INSPECTOR_PROMPT,
            "Here is the ORIGINAL image:",
            types.Part.from_bytes(data=image_to_bytes(original_img), mime_type="image/png"),
            "Here is the GENERATED result:",
            types.Part.from_bytes(data=image_to_bytes(generated_img), mime_type="image/png")
        ]

        response = client.models.generate_content(
            model=MODEL_INSPECTOR,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.0) # 냉철한 판단
        )
        
        if response.text:
            result = response.text.strip()
            if "PASS" in result:
                return True, "PASS"
            else:
                return False, result # 실패 사유 반환
        return True, "Unknown Response (Passed)" # 애매하면 통과
        
    except Exception as e:
        print(f"검수 오류: {e}")
        return True, "Inspector Error (Skipped)" # 검수기 고장나면 그냥 통과

def generate_with_auto_fix(api_key, prompt, image_input, ex_in, ex_out, max_retries=2):
    """
    생성(Worker) -> 검수(Inspector) -> (실패시) 재생성 루프
    Safety Settings를 추가하여 차단율을 낮추고, 검수 피드백을 반영합니다.
    """
    client = genai.Client(api_key=api_key)
    target_bytes = image_to_bytes(image_input)
    
    last_error = ""

    for attempt in range(max_retries + 1):
        try:
            # 1. 콘텐츠 구성
            contents = [prompt]
            
            # 예시 데이터가 있으면 추가 (퓨샷 학습)
            if ex_in and ex_out:
                ex_in_b = image_to_bytes(ex_in)
                ex_out_b = image_to_bytes(ex_out)
                contents.extend([
                    "Example Input Image (Reference):", 
                    types.Part.from_bytes(data=ex_in_b, mime_type="image/png"),
                    "Example Output Image (Target Style):", 
                    types.Part.from_bytes(data=ex_out_b, mime_type="image/png")
                ])
            
            # 이전 시도에서 검수 실패 시 피드백 추가
            if attempt > 0 and last_error:
                contents.append(f"⚠️ PREVIOUS ATTEMPT FAILED: {last_error}")
                contents.append("Please fix the issues mentioned above and try again.")

            # 대상 이미지 추가
            contents.append("Now, process this image:")
            contents.append(types.Part.from_bytes(data=target_bytes, mime_type="image/png"))

            # 2. API 설정 (4K 출력 + 안전 설정 해제)
            config_params = {
                "response_modalities": ["IMAGE"],
                "image_config": types.ImageConfig(image_size="4K")
            }
            
            # 만화의 액션/표현이 차단되지 않도록 모든 카테고리 해제
            safety_settings = [
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            ]

            # 3. 이미지 생성 실행
            response = client.models.generate_content(
                model=MODEL_WORKER,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.2, # 약간의 유연성을 위해 0.2 설정
                    safety_settings=safety_settings,
                    **config_params
                )
            )
            
            result_img = None
            if response.parts:
                for part in response.parts:
                    if part.inline_data: 
                        result_img = Image.open(io.BytesIO(part.inline_data.data))
                    elif hasattr(part, 'image') and part.image: 
                        result_img = part.image
            if not result_img and hasattr(response, 'image') and response.image: 
                result_img = response.image

            if not result_img:
                return None, "이미지 생성 결과가 비어있습니다. (Safety Filter 가능성)"

            # 4. 검수 (Inspector) - 마지막 시도가 아닐 때만 실행
            if attempt < max_retries:
                is_pass, reason = verify_image(api_key, image_input, result_img)
                if is_pass:
                    return result_img, None # 통과 시 즉시 반환
                else:
                    last_error = reason
                    st.toast(f"🚨 검수 불합격 ({attempt+1}/{max_retries}): {reason}")
                    time.sleep(1.5) # API 할당량 제한을 고려한 짧은 대기
                    continue
            else:
                # 마지막 시도라면 검수 결과와 상관없이 출력
                return result_img, "최종 시도 완료 (검수 미통과 포함)"

        except Exception as e:
            # API 에러 발생 시 재시도하지 않고 에러 반환 (Key 문제 등)
            return None, f"API 에러 발생: {str(e)}"
            
    return None, "재시도 횟수를 초과했습니다."

def process_and_update(item, api_key, prompt, ex_in, ex_out, use_autofix):
    with st.spinner(f"✨ 작업 중... ({item['name']})"):
        if use_autofix:
            res_img, err = generate_with_auto_fix(api_key, prompt, item['image'], ex_in, ex_out)
        else:
            # 검수 없이 1회 실행
            res_img, err = generate_with_auto_fix(api_key, prompt, item['image'], ex_in, ex_out, max_retries=0)

        if res_img:
            st.session_state.results.append({'id': str(uuid.uuid4()), 'name': item['name'], 'original': item['image'], 'result': res_img})
            st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]
            save_session_to_disk()
            st.rerun()
        else:
            item['status'] = 'error'
            item['error_msg'] = err
            save_session_to_disk()
            st.rerun()

# --- [4. UI 컴포넌트] ---
def render_sidebar():
    with st.sidebar:
        st.title("🍌 Nano Banana")
        st.caption("Auto-Fix Edition")
        api_key = st.text_input("Google API Key", value=DEFAULT_API_KEY, type="password")
        
        # 모델 선택은 제거 (자동으로 3 Pro + 2 Flash 조합 사용)
        st.info(f"🛠️ 작업자: {MODEL_WORKER}\n👮 감독관: {MODEL_INSPECTOR}")

        st.divider()
        st.subheader("⚙️ 옵션")
        use_autofix = st.toggle("🛡️ 자동 검수 & 재생성", value=True, help="결과물이 이상하면 자동으로 다시 시도합니다. (시간 더 걸림)")
        
        if st.button("🗑️ 초기화", use_container_width=True): clear_all_data()

        st.divider()
        st.subheader("📚 예시 학습")
        ex_in_file = st.file_uploader("예시 원본", type=['png', 'jpg'])
        ex_out_file = st.file_uploader("예시 완성본", type=['png', 'jpg'])
        
        ex_in, ex_out = None, None
        if ex_in_file: ex_in = Image.open(ex_in_file)
        elif os.path.exists(DEFAULT_EX_IN_PATH): ex_in = Image.open(DEFAULT_EX_IN_PATH)
        if ex_out_file: ex_out = Image.open(ex_out_file)
        elif os.path.exists(DEFAULT_EX_OUT_PATH): ex_out = Image.open(DEFAULT_EX_OUT_PATH)

        if ex_in and ex_out: st.success("✅ 예시 적용됨")
        
        st.divider()
        use_slider = st.toggle("비교 슬라이더", value=True)
        with st.expander("📝 프롬프트 수정"):
            prompt = st.text_area("작업 지시사항", value=DEFAULT_PROMPT, height=300)
            
        return api_key, use_slider, prompt, ex_in, ex_out, use_autofix

def handle_file_upload():
    col1, col2 = st.columns([3, 1])
    with col1: files = st.file_uploader("이미지 추가", type=['png', 'jpg', 'zip'], accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")
    with col2:
        st.write("클립보드:")
        paste_btn = paste_image_button(label="📋 붙여넣기", text_color="#ffffff", background_color="#FF4B4B", hover_background_color="#FF0000")

    if files:
        new_cnt = 0
        with st.spinner("파일 읽는 중..."):
            for f in files:
                if f.name.lower().endswith('.zip'):
                    try:
                        with zipfile.ZipFile(f) as z:
                            img_files = [n for n in z.namelist() if n.lower().endswith(('.png','.jpg')) and '__MACOSX' not in n]
                            for fname in img_files:
                                with z.open(fname) as img_f:
                                    st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': os.path.basename(fname), 'image': Image.open(io.BytesIO(img_f.read())), 'status': 'pending', 'error_msg': None})
                                    new_cnt += 1
                    except: pass
                else:
                    try:
                        st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': f.name, 'image': Image.open(f), 'status': 'pending', 'error_msg': None})
                        new_cnt += 1
                    except: pass
            if new_cnt > 0:
                save_session_to_disk()
                time.sleep(0.5)
                st.session_state.uploader_key += 1
                st.rerun()

    if paste_btn.image_data:
        curr_hash = get_image_hash(paste_btn.image_data)
        if st.session_state.last_pasted_hash != curr_hash:
            st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': f"paste_{int(time.time())}.png", 'image': paste_btn.image_data, 'status': 'pending', 'error_msg': None})
            st.session_state.last_pasted_hash = curr_hash
            save_session_to_disk()
            st.rerun()

def render_queue(api_key, prompt, ex_in, ex_out, use_autofix):
    if not st.session_state.job_queue: return

    st.divider()
    c1, c2, c3 = st.columns([3, 1, 1])
    pending = [i for i in st.session_state.job_queue if i['status'] == 'pending']
    c1.subheader(f"📂 대기열 ({len(st.session_state.job_queue)}장)")
    
    if not st.session_state.is_auto_running:
        if c2.button(f"🚀 전체 실행", type="primary", use_container_width=True, disabled=len(pending)==0):
            st.session_state.is_auto_running = True
            st.rerun()
    else:
        if c2.button("⏹️ 중지", type="secondary"):
            st.session_state.is_auto_running = False
            st.rerun()

    if c3.button("🗑️ 선택 삭제"):
        st.session_state.job_queue = []
        save_session_to_disk()
        st.rerun()

    if st.session_state.is_auto_running: st.progress(100, text="🔄 자동 작업 중...")

    for item in st.session_state.job_queue:
        with st.container(border=True):
            col_img, col_info = st.columns([1, 4])
            with col_img:
                st.image(item['image'], use_container_width=True)
                if st.button("🔍 확대", key=f"zoom_q_{item['id']}"): show_full_image(item['image'], item['name'])
            with col_info:
                st.markdown(f"**📄 {item['name']}**")
                if item['status'] == 'error': st.error(f"❌ {item['error_msg']}")
                elif item['status'] == 'pending': st.info("⏳ 대기 중")
                
                b1, b2, b3 = st.columns([1, 1, 3])
                if b1.button("▶️ 실행", key=f"run_{item['id']}"): process_and_update(item, api_key, prompt, ex_in, ex_out, use_autofix)
                if b2.button("🗑️ 삭제", key=f"del_{item['id']}"):
                    st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]
                    save_session_to_disk()
                    st.rerun()

def render_results(use_slider):
    if not st.session_state.results: return

    st.divider()
    c1, c2 = st.columns([4, 1])
    c1.subheader(f"🖼️ 완료 ({len(st.session_state.results)}장)")
    
    if c2.button("🗑️ 비우기"):
        st.session_state.results = []
        save_session_to_disk()
        st.rerun()

    with st.container():
        zip_data = create_zip_file()
        st.download_button("📦 전체 다운로드 (ZIP)", zip_data, "results.zip", "application/zip", use_container_width=True, type="primary")

    st.divider()
    for item in st.session_state.results:
        with st.container(border=True):
            col_img, col_info = st.columns([1, 3])
            with col_img:
                st.image(item['result'], use_container_width=True)
                if st.button("🔍 확대", key=f"zoom_r_{item['id']}"): show_full_image(item['result'], item['name'])
            with col_info:
                st.markdown(f"### ✅ {item['name']}")
                if use_slider:
                    with st.expander("🆚 비교 보기"):
                        orig, res = item['original'], item['result']
                        if orig.size != res.size: orig = orig.resize(res.size)
                        image_comparison(img1=orig, img2=res, label1="Original", label2="Trans", in_memory=True)
                
                cols = st.columns(3)
                if cols[0].button("🔄 재작업", key=f"re_{item['id']}"):
                    st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': item['name'], 'image': item['original'], 'status': 'pending', 'error_msg': None})
                    st.session_state.results = [x for x in st.session_state.results if x['id'] != item['id']]
                    save_session_to_disk()
                    st.rerun()
                if cols[1].button("🗑️ 삭제", key=f"rm_{item['id']}"):
                    st.session_state.results = [x for x in st.session_state.results if x['id'] != item['id']]
                    save_session_to_disk()
                    st.rerun()
                
                buf = io.BytesIO()
                item['result'].save(buf, format="PNG")
                cols[2].download_button("⬇️ 다운", data=buf.getvalue(), file_name=f"kor_{item['name']}", mime="image/png", key=f"dl_{item['id']}")

def auto_process_step(api_key, prompt, ex_in, ex_out, use_autofix):
    if not st.session_state.is_auto_running: return
    pending = [i for i in st.session_state.job_queue if i['status'] == 'pending']
    
    if not pending:
        st.session_state.is_auto_running = False
        st.toast("✅ 작업 완료!")
        time.sleep(1)
        st.rerun()
        return

    item = pending[0]
    with st.spinner(f"자동 처리 중... {item['name']}"):
        if use_autofix:
            res_img, err = generate_with_auto_fix(api_key, prompt, item['image'], ex_in, ex_out)
        else:
            res_img, err = generate_with_auto_fix(api_key, prompt, item['image'], ex_in, ex_out, max_retries=0)

        if res_img:
            st.session_state.results.append({'id': str(uuid.uuid4()), 'name': item['name'], 'original': item['image'], 'result': res_img})
            st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]
            save_session_to_disk()
        else:
            item['status'] = 'error'
            item['error_msg'] = err
            save_session_to_disk()
    
    time.sleep(1)
    st.rerun()

# --- [6. 메인 실행] ---
def main():
    init_session_state()
    api_key, use_slider, prompt, ex_in, ex_out, use_autofix = render_sidebar()
    
    st.title("🍌 Nano Banana")
    st.markdown("**Auto-Fix Edition** (with Supervisor AI)")
    
    handle_file_upload()
    render_queue(api_key, prompt, ex_in, ex_out, use_autofix)
    render_results(use_slider)

    if st.session_state.is_auto_running:
        auto_process_step(api_key, prompt, ex_in, ex_out, use_autofix)

if __name__ == "__main__":
    main()

