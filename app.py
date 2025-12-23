import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
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

# --- [1. 기본 설정 및 상수] ---
DEFAULT_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")

# ✅ 2025 최신 모델 리스트 (사용자 요청 반영)
MODELS = [
    "gemini-3-pro-image-preview",  # 👑 [추천] 식질/화질 끝판왕 (Nano Banana Pro)
    "gemini-2.5-flash-image",      # ⚡ [속도] 가성비 모델 (Nano Banana)
    "gemini-1.5-pro",              # 🛡️ [안전] 구관이 명관 (백업용)
]

RESOLUTIONS = ["원본 유지 (Original)", "1024", "1280", "1920", "2048"]

# ✅ 강력한 식질 프롬프트 (Gemini 3 추론 능력 활용)
DEFAULT_PROMPT = """
# Role
You are the world's best 'Manga Typesetter' and 'Translator', powered by Gemini 3 Pro.

# 1. 🎭 Super-Resolution Translation (초월 번역)
Analyze the characters' emotions, atmosphere, and context deeply.
- **Tone:** If the character is angry, use rough Korean. If shy, use hesitant Korean.
- **Context:** Infer relationships (Senpai/Kohai) and reflect them in honorifics (Jondaemal/Banmal).
- **Naturalness:** Use natural Korean spoken style (Webtoon style).

# 2. 📐 Absolute Layout Rules (가로쓰기 강제)
Readability is King.
- **[CRITICAL] HORIZONTAL TEXT ONLY:** All text MUST be written **Left-to-Right**. Vertical text is strictly FORBIDDEN.
- **Bubble Expansion:** If a speech bubble is too narrow for horizontal text, **EXTEND the white bubble horizontally** (overpaint the background) to fit the text.
- **Line Breaks:** Use frequent line breaks to fit text naturally.

# 3. 🎨 4K In-painting
- Restore the background (screen tones, speed lines) perfectly behind the text.
- Output the image in the **highest possible resolution** (Crisp & Clean).
- **Remove** all original Japanese text completely.

# Output
Return ONLY the edited image file. No JSON, No text.
"""

st.set_page_config(page_title="Nano Banana 3.0", page_icon="🍌", layout="wide")

# --- [2. 초기화 및 유틸리티] ---
def init_session_state():
    defaults = {
        'job_queue': [],
        'results': [],
        'uploader_key': 0,
        'last_pasted_hash': None,
        'is_auto_running': False,
        'allow_mod': True,
        'use_upscale': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_image_hash(image: Image.Image) -> str:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return hashlib.md5(img_byte_arr.getvalue()).hexdigest()

def resize_image_if_needed(image: Image.Image, max_width_setting: str) -> Image.Image:
    if max_width_setting == "원본 유지 (Original)":
        return image
    target_width = int(max_width_setting)
    if image.width > target_width:
        ratio = target_width / float(image.width)
        return image.resize((target_width, int(float(image.height) * ratio)), Image.Resampling.LANCZOS)
    return image

def save_to_local_folder(folder_name):
    if not folder_name:
        st.error("폴더 이름을 입력하세요.")
        return
    try:
        os.makedirs(folder_name, exist_ok=True)
        count = 0
        for item in st.session_state.results:
            safe_name = f"edited_{item['name']}"
            if not safe_name.lower().endswith('.png'):
                safe_name = os.path.splitext(safe_name)[0] + ".png"
            
            save_path = os.path.join(folder_name, safe_name)
            item['result'].save(save_path, format="PNG")
            count += 1
        st.success(f"✅ {count}장 저장 완료: `{os.path.abspath(folder_name)}`")
    except Exception as e:
        st.error(f"저장 실패: {e}")

# --- [3. AI 처리 로직 (Core)] ---

def get_generation_config():
    """
    ✅ [최고 화질 설정]
    - output_tokens: 최대치로 설정하여 고해상도 생성 유도
    - mime_type: 모델이 텍스트가 아닌 이미지를 반환하도록 강제
    """
    return genai.types.GenerationConfig(
        candidate_count=1,
        max_output_tokens=32768, 
        temperature=0.2,
        response_mime_type="image/jpeg" # 이미지 반환 강제
    )

def upscale_with_gemini(api_key: str, image: Image.Image) -> Image.Image:
    """Gemini 3 Pro를 이용한 4K 리마스터링"""
    try:
        genai.configure(api_key=api_key)
        # 업스케일링은 무조건 성능 좋은 3 Pro 사용
        model = genai.GenerativeModel("gemini-3-pro-image-preview") 
        
        prompt = """
        # Task
        **RE-RENDER** this manga page in **4K Ultra-High Resolution**.
        
        # Guidelines
        1. **Denoise & Vectorize:** Remove all JPEG artifacts/noise. Make lines vector-sharp and crisp.
        2. **Preserve Content:** Do NOT change text contents or character designs. Only enhance the visual quality.
        3. **Contrast:** Make blacks deeper and whites brighter (Digital Scan Quality).
        
        # Output
        Return only the high-quality image.
        """
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }

        response = model.generate_content(
            [prompt, image], 
            safety_settings=safety_settings,
            generation_config=get_generation_config()
        )
            
        if hasattr(response, 'parts'):
            for part in response.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    return Image.open(io.BytesIO(part.inline_data.data))
                elif hasattr(part, 'image') and part.image:
                    return part.image
        # 데이터가 없으면 원본 반환
        return image
    except Exception as e:
        print(f"Upscale Fail: {e}")
        return image

def process_single_image(api_key: str, model_name: str, image_input: Image.Image, prompt: str, max_width: str, allow_mod: bool, use_upscale: bool) -> Tuple[Optional[Image.Image], Optional[str]]:
    try:
        processed_input = resize_image_if_needed(image_input, max_width)
        genai.configure(api_key=api_key)
        
        final_prompt = prompt
        if allow_mod:
            final_prompt += """
            \n# 🛠️ [CRITICAL: BUBBLE MODIFICATION]
            If the bubble is too narrow for horizontal text:
            1. **OVERPAINT**: Extend the white background horizontally.
            2. **PRIORITY**: Horizontal Text Readability > Original Bubble Shape.
            """

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
        
        model = genai.GenerativeModel(model_name)
        
        # 1차 생성: 번역 및 식질
        response = model.generate_content(
            [final_prompt, processed_input], 
            safety_settings=safety_settings,
            generation_config=get_generation_config()
        )
        
        result_image = None
        if not response.candidates:
            return None, "AI 응답 거부 (필터/과부하)"
        
        # 이미지 추출
        if hasattr(response, 'parts'):
            for part in response.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    result_image = Image.open(io.BytesIO(part.inline_data.data))
                elif hasattr(part, 'image') and part.image:
                    result_image = part.image
        
        if not result_image:
            return None, "이미지 생성 실패"

        # 2차 생성: 업스케일링 (옵션)
        if use_upscale:
            result_image = upscale_with_gemini(api_key, result_image)

        return result_image, None

    except Exception as e:
        return None, f"API 에러: {str(e)}"

def process_and_update(item, api_key, model, prompt, resolution, allow_bubble_mod, use_upscale):
    """아이템 처리 및 세션 업데이트"""
    msg = "✨ Gemini 3 Pro 리마스터링 중..." if use_upscale else "번역 및 식질 중..."
    with st.spinner(f"{item['name']} 처리 중... ({msg})"):
        res_img, err = process_single_image(api_key, model, item['image'], prompt, resolution, allow_bubble_mod, use_upscale)
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
        st.title("🍌 Nano Banana 3.0")
        st.caption("Powered by Gemini 3 Pro")
        
        api_key = st.text_input("Google API Key", value=DEFAULT_API_KEY, type="password")
        
        model = st.selectbox("모델 선택", MODELS, index=0)
        
        if "gemini-3" in model:
            st.success("🚀 **Gemini 3 Pro**: 4K 지원 & 식질 최강")
        elif "2.5" in model:
            st.info("⚡ **Gemini 2.5 Flash**: 빠른 속도")
            
        st.divider()
        resolution = st.selectbox("최대 너비(Width) 제한", RESOLUTIONS, index=0)
        st.caption("📢 최고 화질을 위해 **'원본 유지'**를 권장합니다.")
        
        st.subheader("🎨 편집 옵션")
        allow_bubble_mod = st.toggle("말풍선 확장/변형 허용", value=True, help="세로 말풍선을 가로 텍스트에 맞춰 강제로 늘립니다.")
        use_upscale = st.toggle("✨ Gemini 3.0 리마스터링 (Upscale)", value=False, help="결과물을 다시 그려서 4K급으로 선명하게 복원합니다.")
        
        st.divider()
        use_slider = st.toggle("비교 슬라이더 사용", value=True)
        prompt = st.text_area("시스템 프롬프트", value=DEFAULT_PROMPT, height=300)
        
        return api_key, model, resolution, use_slider, prompt, allow_bubble_mod, use_upscale

def handle_file_upload():
    col1, col2 = st.columns([3, 1])
    with col1:
        files = st.file_uploader(
            "이미지 또는 ZIP 파일 추가", 
            type=['png', 'jpg', 'jpeg', 'webp', 'zip'], 
            accept_multiple_files=True, 
            key=f"uploader_{st.session_state.uploader_key}"
        )
    with col2:
        st.write("클립보드:")
        paste_btn = paste_image_button(
            label="📋 붙여넣기", text_color="#ffffff", 
            background_color="#FF4B4B", hover_background_color="#FF0000"
        )

    if files:
        new_cnt = 0
        with st.spinner("파일 분석 중..."):
            for f in files:
                if f.name.lower().endswith('.zip'):
                    try:
                        with zipfile.ZipFile(f) as z:
                            img_files = [n for n in z.namelist() if n.lower().endswith(('.png','.jpg','.jpeg','.webp')) and '__MACOSX' not in n]
                            for fname in img_files:
                                with z.open(fname) as img_f:
                                    img = Image.open(io.BytesIO(img_f.read()))
                                    img.load()
                                    st.session_state.job_queue.append({
                                        'id': str(uuid.uuid4()), 'name': os.path.basename(fname), 
                                        'image': img, 'status': 'pending', 'error_msg': None
                                    })
                                    new_cnt += 1
                    except Exception as e:
                        st.error(f"ZIP 오류 ({f.name}): {e}")
                else:
                    try:
                        img = Image.open(f)
                        img.load()
                        st.session_state.job_queue.append({
                            'id': str(uuid.uuid4()), 'name': f.name, 
                            'image': img, 'status': 'pending', 'error_msg': None
                        })
                        new_cnt += 1
                    except:
                        st.toast(f"❌ {f.name} 파일 오류")
            
            if new_cnt > 0:
                time.sleep(0.5)
                st.session_state.uploader_key += 1
                st.rerun()

    if paste_btn.image_data is not None:
        curr_hash = get_image_hash(paste_btn.image_data)
        if st.session_state.last_pasted_hash != curr_hash:
            timestamp = int(time.time())
            st.session_state.job_queue.append({
                'id': str(uuid.uuid4()), 'name': f"clipboard_{timestamp}.png", 
                'image': paste_btn.image_data, 'status': 'pending', 'error_msg': None
            })
            st.session_state.last_pasted_hash = curr_hash
            st.rerun()

def render_queue(api_key, model, prompt, resolution, allow_bubble_mod, use_upscale):
    if not st.session_state.job_queue:
        st.info("대기열이 비어있습니다. 이미지를 업로드하거나 붙여넣으세요.")
        return

    st.divider()
    c1, c2, c3 = st.columns([3, 1, 1])
    pending_count = len([i for i in st.session_state.job_queue if i['status'] == 'pending'])
    c1.subheader(f"📂 대기열 ({len(st.session_state.job_queue)}장 / 대기 {pending_count}장)")
    
    if not st.session_state.is_auto_running:
        if c2.button(f"🚀 전체 실행", type="primary", use_container_width=True, disabled=pending_count==0):
            if not api_key:
                st.error("API 키를 먼저 입력하세요.")
            else:
                st.session_state.is_auto_running = True
                st.rerun()
    else:
        if c2.button("⏹️ 실행 중지", type="secondary", use_container_width=True):
            st.session_state.is_auto_running = False
            st.rerun()
            
    if c3.button("🗑️ 전체 삭제", use_container_width=True):
        st.session_state.job_queue = []
        st.session_state.is_auto_running = False
        st.rerun()

    if st.session_state.is_auto_running:
        st.progress(100, text="🔄 자동 처리 중입니다... (파일을 추가해도 멈추지 않습니다)")

    with st.container():
        for i, item in enumerate(st.session_state.job_queue):
            with st.expander(f"#{i+1} : {item['name']}", expanded=False):
                cols = st.columns([1, 3, 2])
                cols[0].image(item['image'], use_container_width=True)
                with cols[1]:
                    if item['status'] == 'error':
                        st.error(f"❌ {item['error_msg']}")
                    elif item['status'] == 'pending':
                        st.info("⏳ 대기 중")
                    
                with cols[2]:
                    if st.button("▶️ 개별 실행", key=f"run_one_{item['id']}", use_container_width=True):
                         process_and_update(item, api_key, model, prompt, resolution, allow_bubble_mod, use_upscale)
                    
                    if st.button("🗑️ 삭제", key=f"del_q_{item['id']}", use_container_width=True):
                        st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]
                        st.rerun()

def render_results(use_slider):
    if not st.session_state.results:
        return

    st.divider()
    c1, c2 = st.columns([4, 1])
    c1.subheader(f"🖼️ 완료 목록 ({len(st.session_state.results)}장)")
    
    if c2.button("🗑️ 결과 비우기", use_container_width=True):
        st.session_state.results = []
        st.rerun()

    with st.container():
        sc1, sc2 = st.columns([3, 1])
        folder_name = sc1.text_input("폴더명", value="나노바나나_결과물", label_visibility="collapsed", placeholder="저장할 폴더명 입력")
        if sc2.button("💾 폴더에 저장", use_container_width=True):
            save_to_local_folder(folder_name)

    st.divider()
    
    for i, item in enumerate(st.session_state.results):
        with st.expander(f"✅ #{i+1} : {item['name']}", expanded=True):
            cols = st.columns([3, 1])
            with cols[0]:
                if use_slider:
                    orig = item['original']
                    res = item['result']
                    if orig.size != res.size:
                        orig = orig.resize(res.size)
                    
                    image_comparison(
                        img1=orig, img2=res, 
                        label1="Original", label2="Trans",
                        in_memory=True
                    )
                else:
                    st.image(item['result'], use_container_width=True)

            with cols[1]:
                st.caption("작업 관리")
                if st.button("🔄 다시 하기", key=f"retry_res_{item['id']}", help="현재 결과를 삭제하고 대기열로 되돌립니다.", use_container_width=True):
                    st.session_state.job_queue.append({
                        'id': str(uuid.uuid4()), 
                        'name': item['name'], 
                        'image': item['original'], 
                        'status': 'pending', 
                        'error_msg': None
                    })
                    st.session_state.results = [x for x in st.session_state.results if x['id'] != item['id']]
                    st.toast(f"♻️ '{item['name']}' 재작업을 위해 대기열로 이동!", icon="↩️")
                    time.sleep(0.5)
                    st.rerun()

                if st.button("🗑️ 삭제", key=f"del_res_{item['id']}", use_container_width=True):
                    st.session_state.results = [x for x in st.session_state.results if x['id'] != item['id']]
                    st.rerun()
                
                buf = io.BytesIO()
                item['result'].save(buf, format="PNG")
                st.download_button(
                    label="⬇️ 다운로드",
                    data=buf.getvalue(),
                    file_name=f"translated_{item['name']}",
                    mime="image/png",
                    key=f"down_{item['id']}",
                    use_container_width=True
                )

def auto_process_step(api_key, model, prompt, resolution, allow_bubble_mod, use_upscale):
    if not st.session_state.is_auto_running:
        return

    pending_items = [i for i in st.session_state.job_queue if i['status'] == 'pending']
    
    if not pending_items:
        st.session_state.is_auto_running = False
        st.toast("✅ 모든 작업이 완료되었습니다!")
        time.sleep(1)
        st.rerun()
        return

    item = pending_items[0]
    
    msg = "✨ Gemini 3 Pro 리마스터링 중..." if use_upscale else "작업 중..."
    with st.spinner(f"자동 처리 중... {item['name']} ({msg})"):
        res_img, err = process_single_image(api_key, model, item['image'], prompt, resolution, allow_bubble_mod, use_upscale)
        
        if res_img:
            st.session_state.results.append({
                'id': str(uuid.uuid4()), 'name': item['name'], 
                'original': item['image'], 'result': res_img
            })
            st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]
        else:
            item['status'] = 'error'
            item['error_msg'] = err
    
    time.sleep(0.5)
    st.rerun()

# --- [5. 메인 실행] ---
def main():
    init_session_state()
    api_key, model, resolution, use_slider, prompt, allow_bubble_mod, use_upscale = render_sidebar()
    
    st.session_state['allow_mod'] = allow_bubble_mod
    st.session_state['use_upscale'] = use_upscale

    st.title("🍌 Nano Banana 3.0")
    st.markdown("""
    **Ultimate Manga Typesetter powered by Gemini 3 Pro**
    - **Gemini 3 Pro (Nano Banana Pro)**: 4K Resolution & Superior Typesetting
    - **Ultra Upscaling**: Re-render lines with vector-like quality
    """)
    
    handle_file_upload()
    render_queue(api_key, model, prompt, resolution, allow_bubble_mod, use_upscale)
    render_results(use_slider)

    if st.session_state.is_auto_running:
        auto_process_step(api_key, model, prompt, resolution, allow_bubble_mod, use_upscale)

if __name__ == "__main__":
    main()
