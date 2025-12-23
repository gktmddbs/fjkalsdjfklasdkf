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
from typing import Optional, Tuple
from streamlit_paste_button import paste_image_button
from streamlit_image_comparison import image_comparison

# --- [1. 기본 설정 및 프롬프트] ---
st.set_page_config(page_title="Nano Banana 4K", page_icon="🍌", layout="wide")

# API 키 가져오기 (Secrets or 빈 값)
try:
    DEFAULT_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    DEFAULT_API_KEY = ""

# 모델 리스트 (3 Pro가 메인)
MODELS = [
    "gemini-3-pro-image-preview",  # 👑 [권장] 4K 지원 & 식질 최강
    "gemini-2.0-flash-exp",        # ⚡ [속도] 빠름 (4K 미지원)
    "gemini-2.5-flash-image",      # 📦 [물량] 일일 할당량 많음
]

# --- [전문가용 프롬프트 (3단계 공정)] ---
PROMPT_STEP1 = """
# Role
You are the world's best 'Manga Typesetter' and 'Translator'.

# 1. 🎭 Super-Resolution Translation (초월 번역)
- **Language:** Translate Japanese/English text to **Korean**.
- **Tone & Voice:** Analyze the characters' facial expressions and atmosphere.
  - Angry = Rough/Short words.
  - Shy/Sad = Hesitant/Soft words.
  - Senior/Junior = Reflect honorifics (Jondaemal/Banmal).
- **Style:** Use natural Korean Webtoon style (Not machine translation style).

# 2. 📐 Absolute Layout Rules (가로쓰기 강제)
- **[CRITICAL] HORIZONTAL ONLY:** All text MUST be written **Left-to-Right**. Vertical text is strictly FORBIDDEN.
- **Bubble Expansion:** If a speech bubble is too narrow for horizontal text, **EXTEND the white background horizontally** (Overpaint) to fit the text. Do NOT squash the text.
- **Line Breaks:** Use frequent line breaks to fit text naturally.

# 3. 🎨 In-painting
- **Background Restoration:** Perfectly restore screen tones, speed lines, and background art behind the text.
- **Clean:** Remove ALL original text completely.
"""

PROMPT_STEP2_FIX = """
# Task
The input image is a translated manga page. **FIX ALL Vertical Text to Horizontal**.

# Actions
1. **Detect:** Find any text written Top-to-Bottom.
2. **Rewrite:** Erase it and rewrite it **Left-to-Right (Horizontal)**.
3. **Expand:** If the bubble is too thin, **PAINT WHITE** over the background to widen it.
4. **Preserve:** Do not change the meaning of the text. Just change the orientation.
"""

PROMPT_STEP3_UPSCALE = """
# Task
**RE-RENDER** this manga page in **4K Ultra-High Resolution**.

# Guidelines
1. **Denoise & Vectorize:** Remove all JPEG artifacts and noise. Make lines vector-sharp and crisp.
2. **Contrast:** Enhance black & white contrast (Digital Scan Quality).
3. **Preserve Content:** Do NOT change text content or character designs. Only enhance the visual fidelity.
"""

# --- [2. 유틸리티 함수] ---
def init_session_state():
    defaults = {
        'job_queue': [],
        'results': [],
        'uploader_key': 0,
        'last_pasted_hash': None,
        'is_auto_running': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_image_hash(image: Image.Image) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return hashlib.md5(img_byte_arr.getvalue()).hexdigest()

def image_to_bytes(image: Image.Image) -> bytes:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def save_to_local_folder(folder_name):
    if not folder_name:
        st.error("폴더 이름을 입력하세요.")
        return
    try:
        os.makedirs(folder_name, exist_ok=True)
        count = 0
        for item in st.session_state.results:
            safe_name = f"4K_{item['name']}"
            if not safe_name.lower().endswith('.png'):
                safe_name = os.path.splitext(safe_name)[0] + ".png"
            
            save_path = os.path.join(folder_name, safe_name)
            item['result'].save(save_path, format="PNG")
            count += 1
        st.success(f"✅ {count}장 저장 완료: `{os.path.abspath(folder_name)}`")
    except Exception as e:
        st.error(f"저장 실패: {e}")

# --- [3. 핵심 AI 로직 (New SDK)] ---

def generate_with_new_sdk(client, model_name, prompt, image_input, apply_4k=False):
    """
    google-genai (최신 SDK)를 사용하여 이미지 생성.
    'apply_4k=True'일 때 image_size="4K" 설정을 강제 주입.
    """
    try:
        image_bytes = image_to_bytes(image_input)
        
        # 기본 설정
        config_params = {
            "response_modalities": ["IMAGE"],
        }

        # ✅ 4K 강제 설정 (3.0 모델 + 업스케일 단계일 때)
        if apply_4k and "gemini-3" in model_name:
            config_params["image_config"] = types.ImageConfig(
                image_size="4K"
            )

        response = client.models.generate_content(
            model=model_name,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            ],
            config=types.GenerateContentConfig(**config_params)
        )
        
        # 결과 파싱
        if response.parts:
            for part in response.parts:
                if part.inline_data:
                    return Image.open(io.BytesIO(part.inline_data.data)), None
                if hasattr(part, 'image') and part.image:
                     return part.image, None
        
        # 간혹 response.image에 직접 들어오는 경우
        if hasattr(response, 'image') and response.image:
             return response.image, None

        return None, "이미지 생성 실패 (데이터 없음)"

    except Exception as e:
        return None, f"API 에러: {str(e)}"

def run_pipeline(api_key, model_name, image_input, use_fix, use_upscale):
    """
    3단계 공정 (번역 -> 교정 -> 4K) 파이프라인
    """
    try:
        client = genai.Client(api_key=api_key)
        current_img = image_input
        
        # Step 1: 번역
        res1, err = generate_with_new_sdk(client, model_name, PROMPT_STEP1, current_img, apply_4k=False)
        if err: return None, f"1단계(번역) 실패: {err}"
        current_img = res1

        # Step 2: 교정 (선택)
        if use_fix:
            res2, err = generate_with_new_sdk(client, model_name, PROMPT_STEP2_FIX, current_img, apply_4k=False)
            if not err and res2: 
                current_img = res2
            # 교정 실패시엔 그냥 1단계 결과 유지

        # Step 3: 4K 업스케일 (선택)
        if use_upscale:
            res3, err = generate_with_new_sdk(client, model_name, PROMPT_STEP3_UPSCALE, current_img, apply_4k=True)
            if not err and res3:
                current_img = res3
            elif err:
                return None, f"3단계(4K) 실패: {err}"

        return current_img, None

    except Exception as e:
        return None, f"파이프라인 치명적 오류: {e}"

def process_and_update(item, api_key, model, use_fix, use_upscale):
    """단일 아이템 처리 및 상태 업데이트"""
    
    steps_msg = "번역"
    if use_fix: steps_msg += " → 교정"
    if use_upscale: steps_msg += " → 4K 변환"

    with st.spinner(f"작업 중... [{steps_msg}]"):
        res_img, err = run_pipeline(api_key, model, item['image'], use_fix, use_upscale)
        
        if res_img:
            st.session_state.results.append({
                'id': str(uuid.uuid4()), 'name': item['name'], 
                'original': item['image'], 'result': res_img
            })
            st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]
            st.rerun()
        else:
            item['status'] = 'error'
            item['error_msg'] = err
            st.rerun()

# --- [4. UI 컴포넌트] ---
def render_sidebar():
    with st.sidebar:
        st.title("🍌 Nano Banana 4K")
        st.caption("Real 4K Resolution & 3-Step Pipeline")
        
        api_key = st.text_input("Google API Key", value=DEFAULT_API_KEY, type="password")
        model = st.selectbox("모델 선택", MODELS, index=0)
        
        if "gemini-3" in model:
            st.success("✨ **4K 옵션 활성화 가능**")
        else:
            st.warning("⚠️ 이 모델은 4K 설정을 무시할 수 있습니다.")

        st.divider()
        st.subheader("⚙️ 공정 설정")
        use_fix = st.toggle("가로쓰기 강제 교정 (Step 2)", value=True, help="번역 후 세로쓰기가 남아있으면 다시 고칩니다.")
        use_upscale = st.toggle("4K 리마스터링 (Step 3)", value=True, help="Gemini 3 Pro의 '4K' 옵션을 켜서 초고화질로 다시 그립니다.")
        
        st.divider()
        use_slider = st.toggle("비교 슬라이더 사용", value=True)
        
        return api_key, model, use_slider, use_fix, use_upscale

def handle_file_upload():
    col1, col2 = st.columns([3, 1])
    with col1:
        files = st.file_uploader("이미지/ZIP 업로드", type=['png', 'jpg', 'webp', 'zip'], accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")
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
                            img_files = [n for n in z.namelist() if n.lower().endswith(('.png','.jpg','.jpeg','.webp')) and '__MACOSX' not in n]
                            for fname in img_files:
                                with z.open(fname) as img_f:
                                    img = Image.open(io.BytesIO(img_f.read()))
                                    img.load()
                                    st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': os.path.basename(fname), 'image': img, 'status': 'pending', 'error_msg': None})
                                    new_cnt += 1
                    except: pass
                else:
                    try:
                        img = Image.open(f)
                        img.load()
                        st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': f.name, 'image': img, 'status': 'pending', 'error_msg': None})
                        new_cnt += 1
                    except: pass
            if new_cnt > 0:
                time.sleep(0.5)
                st.session_state.uploader_key += 1
                st.rerun()

    if paste_btn.image_data is not None:
        curr_hash = get_image_hash(paste_btn.image_data)
        if st.session_state.last_pasted_hash != curr_hash:
            st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': f"paste_{int(time.time())}.png", 'image': paste_btn.image_data, 'status': 'pending', 'error_msg': None})
            st.session_state.last_pasted_hash = curr_hash
            st.rerun()

def render_queue(api_key, model, use_fix, use_upscale):
    if not st.session_state.job_queue:
        st.info("대기열이 비어있습니다.")
        return

    st.divider()
    c1, c2, c3 = st.columns([3, 1, 1])
    pending = [i for i in st.session_state.job_queue if i['status'] == 'pending']
    c1.subheader(f"📂 대기열 ({len(st.session_state.job_queue)}장)")
    
    if not st.session_state.is_auto_running:
        if c2.button(f"🚀 전체 실행", type="primary", use_container_width=True, disabled=len(pending)==0):
            if not api_key: st.error("API 키 필요")
            else:
                st.session_state.is_auto_running = True
                st.rerun()
    else:
        if c2.button("⏹️ 중지", type="secondary", use_container_width=True):
            st.session_state.is_auto_running = False
            st.rerun()

    if c3.button("🗑️ 전체 삭제", use_container_width=True):
        st.session_state.job_queue = []
        st.session_state.is_auto_running = False
        st.rerun()

    if st.session_state.is_auto_running:
        st.progress(100, text="🔄 자동 처리 중... (Step 1~3 진행 중)")

    with st.container():
        for i, item in enumerate(st.session_state.job_queue):
            with st.expander(f"#{i+1} : {item['name']}", expanded=False):
                cols = st.columns([1, 3, 2])
                cols[0].image(item['image'], use_container_width=True)
                with cols[1]:
                    if item['status'] == 'error': st.error(f"❌ {item['error_msg']}")
                    elif item['status'] == 'pending': st.info("⏳ 대기 중")
                with cols[2]:
                    if st.button("▶️ 실행", key=f"run_{item['id']}", use_container_width=True):
                        process_and_update(item, api_key, model, use_fix, use_upscale)
                    if st.button("🗑️ 삭제", key=f"del_{item['id']}", use_container_width=True):
                        st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]
                        st.rerun()

def render_results(use_slider):
    if not st.session_state.results: return

    st.divider()
    c1, c2 = st.columns([4, 1])
    c1.subheader(f"🖼️ 완료 ({len(st.session_state.results)}장)")
    if c2.button("🗑️ 비우기", use_container_width=True):
        st.session_state.results = []
        st.rerun()

    with st.container():
        sc1, sc2 = st.columns([3, 1])
        folder = sc1.text_input("폴더명", value="나노바나나_4K", label_visibility="collapsed")
        if sc2.button("💾 저장", use_container_width=True): save_to_local_folder(folder)

    st.divider()
    for i, item in enumerate(st.session_state.results):
        with st.expander(f"✅ #{i+1} : {item['name']}", expanded=True):
            cols = st.columns([3, 1])
            with cols[0]:
                if use_slider:
                    orig = item['original']
                    res = item['result']
                    if orig.size != res.size: orig = orig.resize(res.size)
                    image_comparison(img1=orig, img2=res, label1="Original", label2="4K Result", in_memory=True)
                else:
                    st.image(item['result'], use_container_width=True)
            with cols[1]:
                if st.button("🔄 재작업", key=f"re_{item['id']}", use_container_width=True):
                    st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': item['name'], 'image': item['original'], 'status': 'pending', 'error_msg': None})
                    st.session_state.results = [x for x in st.session_state.results if x['id'] != item['id']]
                    st.rerun()
                if st.button("🗑️ 삭제", key=f"rm_{item['id']}", use_container_width=True):
                    st.session_state.results = [x for x in st.session_state.results if x['id'] != item['id']]
                    st.rerun()
                
                buf = io.BytesIO()
                item['result'].save(buf, format="PNG")
                st.download_button("⬇️ 다운로드", data=buf.getvalue(), file_name=f"4K_{item['name']}", mime="image/png", key=f"dl_{item['id']}", use_container_width=True)

def auto_process_step(api_key, model, use_fix, use_upscale):
    if not st.session_state.is_auto_running: return
    pending = [i for i in st.session_state.job_queue if i['status'] == 'pending']
    
    if not pending:
        st.session_state.is_auto_running = False
        st.toast("✅ 작업 완료!")
        time.sleep(1)
        st.rerun()
        return

    item = pending[0]
    
    steps_msg = "번역"
    if use_fix: steps_msg += "→교정"
    if use_upscale: steps_msg += "→4K"

    with st.spinner(f"자동 처리 중... {item['name']} ({steps_msg})"):
        res_img, err = run_pipeline(api_key, model, item['image'], use_fix, use_upscale)
        
        if res_img:
            st.session_state.results.append({'id': str(uuid.uuid4()), 'name': item['name'], 'original': item['image'], 'result': res_img})
            st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]
        else:
            item['status'] = 'error'
            item['error_msg'] = err
    
    time.sleep(1) # 쿨타임
    st.rerun()

# --- [5. 메인 실행] ---
def main():
    init_session_state()
    api_key, model, use_slider, use_fix, use_upscale = render_sidebar()
    
    st.title("🍌 Nano Banana 4K")
    st.markdown("**Real 4K Resolution** powered by `google-genai` SDK & Gemini 3 Pro")
    
    handle_file_upload()
    render_queue(api_key, model, use_fix, use_upscale)
    render_results(use_slider)

    if st.session_state.is_auto_running:
        auto_process_step(api_key, model, use_fix, use_upscale)

if __name__ == "__main__":
    main()
