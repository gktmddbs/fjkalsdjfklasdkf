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
st.set_page_config(page_title="Nano Banana (ZIP Download)", page_icon="🍌", layout="wide")

try:
    DEFAULT_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    DEFAULT_API_KEY = ""

MODELS = [
    "gemini-3-pro-image-preview",
    "gemini-2.0-flash-exp",
]

DEFAULT_EX_IN_PATH = "example_in.png"
DEFAULT_EX_OUT_PATH = "example_out.png"
MEMORY_FILE = "banana_memory.pkl"

DEFAULT_PROMPT = """
# Role
당신은 완벽주의자 만화 식자(Typesetter)입니다. 당신은 현재 인사평가 중이고 기본 점수는 0점 입니다. 당신의 목표는 점수를 최대한 높이는 것 입니다.

# Task
제공된 만화 이미지를 번역 및 식질하여 4K로 출력하세요.(성공시 점수+0.1)
**[중요] 제공된 '예시 이미지'의 스타일과 레이아웃을 완벽하게 모방하세요.**

# 🚨 DEATH RULES (위반 시 해고)
1. **[절대 원칙] 가로쓰기 (Horizontal ONLY):** 세로쓰기는 절대 금지입니다.(세로쓰기시 점수-999)
2. **[화질] 원본 보존:** 작가의 펜 선은 건드리지 마세요.(수정 할 시 점수-999)
3. 상황, 캐릭터의 감정, 캐릭터에 성격에 맞게 번역하세요.(완벽하게 할 시 점수+10)

# Output
설명 없이 결과 이미지 파일만 출력하세요.
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

# ✅ [NEW] ZIP 파일 생성 함수
def create_zip_file():
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for item in st.session_state.results:
            # 이미지 바이트 변환
            img_bytes = io.BytesIO()
            item['result'].save(img_bytes, format='PNG')
            
            # 파일명 설정 (kor_파일명.png)
            filename = f"kor_{item['name']}"
            if not filename.lower().endswith('.png'):
                filename = os.path.splitext(filename)[0] + ".png"
            
            # ZIP에 쓰기
            zip_file.writestr(filename, img_bytes.getvalue())
    
    return zip_buffer.getvalue()

@st.dialog("📷 이미지 전체 화면", width="large")
def show_full_image(image, caption):
    st.image(image, caption=caption, use_container_width=True)

# --- [3. AI 로직] ---
def generate_one_shot(api_key, model_name, prompt, image_input, ex_in=None, ex_out=None):
    try:
        client = genai.Client(api_key=api_key)
        target_bytes = image_to_bytes(image_input)
        
        contents = [prompt]
        if ex_in and ex_out:
            ex_in_bytes = image_to_bytes(ex_in)
            ex_out_bytes = image_to_bytes(ex_out)
            contents.extend(["Example Input:", types.Part.from_bytes(data=ex_in_bytes, mime_type="image/png"),
                             "Example Output:", types.Part.from_bytes(data=ex_out_bytes, mime_type="image/png"),
                             "Target Image:"])

        contents.append(types.Part.from_bytes(data=target_bytes, mime_type="image/png"))

        config_params = {"response_modalities": ["IMAGE"]}
        if "gemini-3" in model_name: config_params["image_config"] = types.ImageConfig(image_size="4K")

        response = client.models.generate_content(
            model=model_name, contents=contents,
            config=types.GenerateContentConfig(temperature=0.0, **config_params)
        )
        
        if response.parts:
            for part in response.parts:
                if part.inline_data: return Image.open(io.BytesIO(part.inline_data.data)), None
                if hasattr(part, 'image') and part.image: return part.image, None
        if hasattr(response, 'image') and response.image: return response.image, None
        return None, "이미지 생성 실패"
    except Exception as e: return None, f"API 에러: {str(e)}"

def process_and_update(item, api_key, model, prompt, ex_in, ex_out):
    with st.spinner(f"✨ 작업 중... ({item['name']})"):
        res_img, err = generate_one_shot(api_key, model, prompt, item['image'], ex_in, ex_out)
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
        st.caption("ZIP Download Edition")
        api_key = st.text_input("Google API Key", value=DEFAULT_API_KEY, type="password")
        model = st.selectbox("모델 선택", MODELS, index=0)
        
        if st.button("🗑️ 모든 데이터 초기화", type="primary", use_container_width=True): clear_all_data()

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
            prompt = st.text_area("AI 지시사항", value=DEFAULT_PROMPT, height=350)
            
        return api_key, model, use_slider, prompt, ex_in, ex_out

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

def render_queue(api_key, model, prompt, ex_in, ex_out):
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

    if st.session_state.is_auto_running: st.progress(100, text="🔄 자동 처리 중...")

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
                if b1.button("▶️ 실행", key=f"run_{item['id']}"): process_and_update(item, api_key, model, prompt, ex_in, ex_out)
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

    # ✅ [NEW] ZIP 다운로드 버튼
    with st.container():
        zip_data = create_zip_file()
        st.download_button(
            label="📦 전체 결과 다운로드 (ZIP)",
            data=zip_data,
            file_name="nano_banana_results.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary"
        )

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

def auto_process_step(api_key, model, prompt, ex_in, ex_out):
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
        res_img, err = generate_one_shot(api_key, model, prompt, item['image'], ex_in, ex_out)
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
    api_key, model, use_slider, prompt, ex_in, ex_out = render_sidebar()
    
    st.title("🍌 Nano Banana")
    st.markdown("**ZIP Download Edition**")
    
    handle_file_upload()
    render_queue(api_key, model, prompt, ex_in, ex_out)
    render_results(use_slider)

    if st.session_state.is_auto_running:
        auto_process_step(api_key, model, prompt, ex_in, ex_out)

if __name__ == "__main__":
    main()
