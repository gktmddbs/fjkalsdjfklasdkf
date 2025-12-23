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

# --- [1. 기본 설정] ---
st.set_page_config(page_title="Nano Banana 4K", page_icon="🍌", layout="wide")

# API 키 가져오기
try:
    DEFAULT_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    DEFAULT_API_KEY = ""

# 모델 리스트
MODELS = [
    "gemini-3-pro-image-preview",  # 👑 4K 지원 & 식질 최강
    "gemini-2.0-flash-exp",        # ⚡ 빠름 (4K 미지원)
    "gemini-2.5-flash-image",      # 📦 물량 많음
]

# --- [2. 한국어 기본 프롬프트 설정] ---
# 사용자가 수정 가능하도록 변수로 분리했습니다.

DEFAULT_PROMPT_STEP1 = """
# Role
당신은 세계 최고의 만화 번역가이자 식자(Typesetter)입니다.

# Task
1. 이미지 내의 일본어/영어를 **한국어**로 번역하세요.
   - 문맥과 캐릭터의 표정을 파악하여 자연스러운 웹툰체로 번역하세요.
2. **[중요] 가로쓰기 강제:** 모든 텍스트는 반드시 **왼쪽에서 오른쪽(가로)**으로 쓰세요. 세로쓰기는 절대 금지입니다.
3. **식질(In-painting):** 글자를 지운 배경(스크린톤, 효과선)을 위화감 없이 완벽하게 복원하세요.
4. **Clean:** 원본 글자는 깨끗하게 지우세요.
"""

DEFAULT_PROMPT_STEP2 = """
# Task
방금 번역된 만화 이미지의 레이아웃을 교정하세요.

# Actions
1. **세로쓰기 감지:** 위에서 아래로(세로로) 써진 텍스트를 찾으세요.
2. **가로로 다시 쓰기:** 해당 텍스트를 지우고, **왼쪽에서 오른쪽(가로)** 방향으로 다시 쓰세요.
3. **말풍선 확장:** 가로로 쓸 공간이 부족하다면, 말풍선 배경을 하얗게 칠해서 옆으로 넓히세요. (글자를 찌그러뜨리지 마세요)
"""

DEFAULT_PROMPT_STEP3 = """
# Task
이 이미지를 **4K 초고해상도**로 다시 렌더링(Re-render)하세요.

# Guidelines
1. **화질 개선:** 노이즈를 제거하고 선을 벡터처럼 선명하게 만드세요.
2. **명암비:** 흑백 명암을 뚜렷하게 보정하세요 (디지털 스캔 품질).
3. **보존:** 텍스트의 내용이나 캐릭터의 생김새는 절대 바꾸지 마세요. 오직 화질만 높이세요.
"""

# --- [3. 유틸리티 함수] ---
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

# --- [4. AI 생성 로직 (New SDK)] ---

def generate_with_new_sdk(client, model_name, prompt, image_input, apply_4k=False):
    """최신 SDK 사용 생성 함수"""
    try:
        image_bytes = image_to_bytes(image_input)
        
        config_params = {
            "response_modalities": ["IMAGE"],
        }

        # 4K 옵션 적용 (3 Pro 모델 & 업스케일 단계일 때)
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
        
        if response.parts:
            for part in response.parts:
                if part.inline_data:
                    return Image.open(io.BytesIO(part.inline_data.data)), None
                if hasattr(part, 'image') and part.image:
                     return part.image, None
        
        if hasattr(response, 'image') and response.image:
             return response.image, None

        return None, "이미지 생성 실패 (데이터 없음)"

    except Exception as e:
        return None, f"API 에러: {str(e)}"

def run_pipeline(api_key, model_name, image_input, use_fix, use_upscale, p1, p2, p3):
    """
    사용자 정의 프롬프트(p1, p2, p3)를 받아 실행하는 파이프라인
    """
    try:
        client = genai.Client(api_key=api_key)
        current_img = image_input
        
        # Step 1: 번역
        res1, err = generate_with_new_sdk(client, model_name, p1, current_img, apply_4k=False)
        if err: return None, f"1단계 실패: {err}"
        current_img = res1

        # Step 2: 교정
        if use_fix:
            res2, err = generate_with_new_sdk(client, model_name, p2, current_img, apply_4k=False)
            if not err and res2: current_img = res2

        # Step 3: 4K 업스케일
        if use_upscale:
            res3, err = generate_with_new_sdk(client, model_name, p3, current_img, apply_4k=True)
            if not err and res3:
                current_img = res3
            elif err:
                return None, f"3단계 실패: {err}"

        return current_img, None

    except Exception as e:
        return None, f"파이프라인 오류: {e}"

def process_and_update(item, api_key, model, use_fix, use_upscale, p1, p2, p3):
    steps_msg = "번역"
    if use_fix: steps_msg += " → 교정"
    if use_upscale: steps_msg += " → 4K 변환"

    with st.spinner(f"작업 중... [{steps_msg}]"):
        # 프롬프트 전달
        res_img, err = run_pipeline(api_key, model, item['image'], use_fix, use_upscale, p1, p2, p3)
        
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

# --- [5. UI 컴포넌트] ---
def render_sidebar():
    with st.sidebar:
        st.title("🍌 Nano Banana 4K")
        st.caption("Real 4K & Custom Prompts")
        
        api_key = st.text_input("Google API Key", value=DEFAULT_API_KEY, type="password")
        model = st.selectbox("모델 선택", MODELS, index=0)
        
        if "gemini-3" in model:
            st.success("✨ **4K 옵션 지원됨**")
        else:
            st.warning("⚠️ 이 모델은 4K 설정을 무시할 수 있습니다.")

        st.divider()
        st.subheader("⚙️ 공정 설정")
        use_fix = st.toggle("가로쓰기 강제 교정 (Step 2)", value=True)
        use_upscale = st.toggle("4K 리마스터링 (Step 3)", value=True)
        
        # --- [프롬프트 커스텀 영역] ---
        st.divider()
        with st.expander("📝 프롬프트 설정 (한국어)", expanded=False):
            st.caption("AI에게 내릴 지시사항을 직접 수정하세요.")
            p1 = st.text_area("Step 1 (번역/식질)", value=DEFAULT_PROMPT_STEP1, height=200)
            p2 = st.text_area("Step 2 (레이아웃 교정)", value=DEFAULT_PROMPT_STEP2, height=150)
            p3 = st.text_area("Step 3 (4K 업스케일)", value=DEFAULT_PROMPT_STEP3, height=150)
        # ---------------------------

        st.divider()
        use_slider = st.toggle("비교 슬라이더 사용", value=True)
        
        return api_key, model, use_slider, use_fix, use_upscale, p1, p2, p3

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

def render_queue(api_key, model, use_fix, use_upscale, p1, p2, p3):
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
        st.progress(100, text="🔄 자동 처리 중... (1~3단계)")

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
                        process_and_update(item, api_key, model, use_fix, use_upscale, p1, p2, p3)
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

def auto_process_step(api_key, model, use_fix, use_upscale, p1, p2, p3):
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
        res_img, err = run_pipeline(api_key, model, item['image'], use_fix, use_upscale, p1, p2, p3)
        
        if res_img:
            st.session_state.results.append({'id': str(uuid.uuid4()), 'name': item['name'], 'original': item['image'], 'result': res_img})
            st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]
        else:
            item['status'] = 'error'
            item['error_msg'] = err
    
    time.sleep(1)
    st.rerun()

# --- [6. 메인 실행] ---
def main():
    init_session_state()
    # 사이드바에서 프롬프트 값(p1, p2, p3)을 받아옴
    api_key, model, use_slider, use_fix, use_upscale, p1, p2, p3 = render_sidebar()
    
    st.title("🍌 Nano Banana 4K")
    st.markdown("**Real 4K & Custom Prompt Edition**")
    
    handle_file_upload()
    # 프롬프트 값을 렌더링 함수에 전달
    render_queue(api_key, model, use_fix, use_upscale, p1, p2, p3)
    render_results(use_slider)

    if st.session_state.is_auto_running:
        auto_process_step(api_key, model, use_fix, use_upscale, p1, p2, p3)

if __name__ == "__main__":
    main()
