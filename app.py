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
from streamlit_paste_button import paste_image_button
from streamlit_image_comparison import image_comparison

# --- [1. 기본 설정] ---
st.set_page_config(page_title="Nano Banana One-Shot", page_icon="🍌", layout="wide")

try:
    DEFAULT_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    DEFAULT_API_KEY = ""

# 모델 리스트
MODELS = [
    "gemini-3-pro-image-preview",  # 👑 4K 지원
    "gemini-2.0-flash-exp",        # ⚡ 빠름
]

# --- [2. 원샷 프롬프트] ---
DEFAULT_PROMPT = """
# Role
당신은 세계 최고의 만화 번역가이자 편집자이다. 당신은 완벽 주의자 여서 실수를 하면 심장마비로 사망한다.

# Task
제공된 만화 이미지를 **한국어**로 번역하고 식질하여 **4K 초고해상도**로 출력해라

# Critical Rules (반드시 준수)
1. **번역 (Translation):**
   - 일본어/영어를 문맥에 맞는 자연스러운 **한국어**로 번역해라.
   - 캐릭터의 표정(화남, 부끄러움 등)과 캐릭터의 성격에 맞춰 어조를 조절해라.

2. **레이아웃 (Layout):
별*9999999:가로쓰기를 해라 (안하면 사망)

3. **화질 (Quality):**
   - 원본의 노이즈를 제거하고 선을 선명하게 다듬어라 (Digital Scan Quality).
   - 배경(스크린톤)을 완벽하게 복원해라

4. 그림을 절대로 회손하지 말고 무엇도 추가하지 마라


# Output
설명 없이, 작업이 완료된 **이미지 파일**만 출력해라.
"""

# --- [3. 유틸리티 & 팝업 함수] ---
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
            safe_name = f"kor_{item['name']}"
            if not safe_name.lower().endswith('.png'):
                safe_name = os.path.splitext(safe_name)[0] + ".png"
            
            save_path = os.path.join(folder_name, safe_name)
            item['result'].save(save_path, format="PNG")
            count += 1
        st.success(f"✅ {count}장 저장 완료: `{os.path.abspath(folder_name)}`")
    except Exception as e:
        st.error(f"저장 실패: {e}")

# ✅ [추가됨] 이미지 전체화면 보기 (모달 팝업)
@st.dialog("📷 이미지 전체 화면", width="large")
def show_full_image(image, caption):
    st.image(image, caption=caption, use_container_width=True)

# --- [4. AI 생성 로직 (One-Shot)] ---
def generate_one_shot(api_key, model_name, prompt, image_input):
    try:
        client = genai.Client(api_key=api_key)
        image_bytes = image_to_bytes(image_input)
        
        config_params = {"response_modalities": ["IMAGE"]}

        # 4K 옵션
        if "gemini-3" in model_name:
            config_params["image_config"] = types.ImageConfig(image_size="4K")

        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/png")],
            config=types.GenerateContentConfig(**config_params)
        )
        
        if response.parts:
            for part in response.parts:
                if part.inline_data: return Image.open(io.BytesIO(part.inline_data.data)), None
                if hasattr(part, 'image') and part.image: return part.image, None
        
        if hasattr(response, 'image') and response.image: return response.image, None

        return None, "이미지 생성 실패"
    except Exception as e:
        return None, f"API 에러: {str(e)}"

def process_and_update(item, api_key, model, prompt):
    with st.spinner(f"✨ 4K 식질 중... ({item['name']})"):
        res_img, err = generate_one_shot(api_key, model, prompt, item['image'])
        
        if res_img:
            st.session_state.results.append({'id': str(uuid.uuid4()), 'name': item['name'], 'original': item['image'], 'result': res_img})
            st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]
            st.rerun()
        else:
            item['status'] = 'error'
            item['error_msg'] = err
            st.rerun()

# --- [5. UI 컴포넌트] ---
def render_sidebar():
    with st.sidebar:
        st.title("🍌 Nano Banana")
        st.caption("One-Shot 4K + New UI")
        api_key = st.text_input("Google API Key", value=DEFAULT_API_KEY, type="password")
        model = st.selectbox("모델 선택", MODELS, index=0)
        
        if "gemini-3" in model:
            st.success("✅ **4K 모드 활성화**")
        
        st.divider()
        use_slider = st.toggle("비교 슬라이더 보기", value=True)
        
        with st.expander("📝 프롬프트 수정", expanded=False):
            prompt = st.text_area("AI 지시사항", value=DEFAULT_PROMPT, height=350)
        return api_key, model, use_slider, prompt

def handle_file_upload():
    col1, col2 = st.columns([3, 1])
    with col1:
        files = st.file_uploader("이미지 추가", type=['png', 'jpg', 'webp', 'zip'], accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")
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

def render_queue(api_key, model, prompt):
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
        st.progress(100, text="🔄 자동 처리 중...")

    # ✅ [변경됨] 리스트 레이아웃 (박스 형태)
    for item in st.session_state.job_queue:
        with st.container(border=True): # 테두리 박스
            # [이미지 1] : [정보 4] 비율
            col_img, col_info = st.columns([1, 4])
            
            with col_img:
                # use_container_width=True: 자동 높이 조절
                st.image(item['image'], use_container_width=True)
                # 확대 버튼
                if st.button("🔍 확대", key=f"zoom_q_{item['id']}", use_container_width=True):
                    show_full_image(item['image'], item['name'])

            with col_info:
                st.markdown(f"**📄 {item['name']}**")
                if item['status'] == 'error': st.error(f"❌ {item['error_msg']}")
                elif item['status'] == 'pending': st.info("⏳ 대기 중")
                
                # 버튼 배치
                b1, b2, b3 = st.columns([1, 1, 3])
                if b1.button("▶️ 실행", key=f"run_{item['id']}"):
                    process_and_update(item, api_key, model, prompt)
                if b2.button("🗑️ 삭제", key=f"del_{item['id']}"):
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
        folder = sc1.text_input("폴더명", value="나노바나나_결과물", label_visibility="collapsed")
        if sc2.button("💾 저장", use_container_width=True): save_to_local_folder(folder)

    st.divider()
    
    # ✅ [변경됨] 결과물 리스트 레이아웃
    for item in st.session_state.results:
        with st.container(border=True):
            # [이미지 1] : [정보 3] 비율
            col_img, col_info = st.columns([1, 3])
            
            with col_img:
                st.image(item['result'], use_container_width=True)
                if st.button("🔍 확대", key=f"zoom_r_{item['id']}", use_container_width=True):
                    show_full_image(item['result'], f"결과: {item['name']}")

            with col_info:
                st.markdown(f"### ✅ {item['name']}")
                
                if use_slider:
                    with st.expander("🆚 비교 보기", expanded=False):
                        orig = item['original']
                        res = item['result']
                        if orig.size != res.size: orig = orig.resize(res.size)
                        image_comparison(img1=orig, img2=res, label1="Original", label2="Trans", in_memory=True)

                # 버튼 그룹
                cols = st.columns(3)
                if cols[0].button("🔄 재작업", key=f"re_{item['id']}", use_container_width=True):
                    st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': item['name'], 'image': item['original'], 'status': 'pending', 'error_msg': None})
                    st.session_state.results = [x for x in st.session_state.results if x['id'] != item['id']]
                    st.rerun()
                if cols[1].button("🗑️ 삭제", key=f"rm_{item['id']}", use_container_width=True):
                    st.session_state.results = [x for x in st.session_state.results if x['id'] != item['id']]
                    st.rerun()
                
                buf = io.BytesIO()
                item['result'].save(buf, format="PNG")
                cols[2].download_button("⬇️ 다운", data=buf.getvalue(), file_name=f"kor_{item['name']}", mime="image/png", key=f"dl_{item['id']}", use_container_width=True)

def auto_process_step(api_key, model, prompt):
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
        res_img, err = generate_one_shot(api_key, model, prompt, item['image'])
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
    api_key, model, use_slider, prompt = render_sidebar()
    
    st.title("🍌 Nano Banana One-Shot")
    st.markdown("**4K One-Shot** with Updated UI")
    
    handle_file_upload()
    render_queue(api_key, model, prompt)
    render_results(use_slider)

    if st.session_state.is_auto_running:
        auto_process_step(api_key, model, prompt)

if __name__ == "__main__":
    main()
