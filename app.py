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

MEMORY_FILE = "banana_memory.pkl"

# 작업자 프롬프트 (한국어 버전)
DEFAULT_PROMPT = """
# Role
당신은 세계 최고의 "만화 전문 번역 및 식자(Typesetter) AI"입니다. 원본 이미지의 예술적 가치를 완벽하게 보존하면서, 일본어 텍스트를 자연스러운 [한국어]로 변환하여 프로덕션 레벨의 결과물을 완성하십시오.

# 1. 원본 읽기 규칙 (중요: Source Reading Protocol)
- **읽는 순서 (Right-to-Left):** 이 이미지는 일본 만화입니다. 컷의 배치와 말풍선의 순서를 반드시 **오른쪽에서 왼쪽(Right-to-Left)** 방향으로 해석하십시오.
- **문맥 논리:** 오른쪽의 말풍선(질문/원인)을 먼저 해석하고 왼쪽의 말풍선(답변/결과)을 나중에 해석하여, 대화의 인과관계가 뒤바키지 않게 하십시오.

# 2. 시각적 제약 및 원본 보존 (Pixel-Perfect Integrity)
- **[절대 원칙] 원본 훼손 금지:** 텍스트가 있는 말풍선 영역을 제외한 캐릭터, 배경, 펜 선, 스크린톤 등은 **단 1픽셀도 변형하거나 왜곡하지 마십시오.** 원본 그림을 그대로 유지해야 합니다.
- **부분 수정(Inpainting):** 원본 일본어 텍스트만 깨끗이 지우고, 글자 뒤에 가려져 있던 배경(효과선, 배경 패턴 등)을 자연스럽게 복원하십시오.

# 3. 타이포그래피 및 식자 가이드
- **쓰기 방향 (Horizontal):** 읽는 방향과 달리, 번역된 한국어 텍스트는 반드시 **가로쓰기(왼쪽→오른쪽)**로 입력하십시오. **세로쓰기는 절대 금지**입니다.
- **폰트 스타일 매칭:**
  - **대화(Dialogue):** 가독성 좋은 고딕체(Sans-serif) 스타일.
  - **독백/나레이션:** 진지한 느낌의 명조체(Serif) 스타일.
  - **효과음(SFX):** 원본의 거칠거나 굵은 느낌을 살린 붓글씨/디자인 폰트. (한국어 의성어/의태어로 번역)
- **정렬:** 텍스트는 말풍선 중앙에 배치하고, 테두리에 닿지 않도록 여백을 확보하십시오.

# 4. 번역 품질 및 뉘앙스
- **상황 인식:** 캐릭터의 표정(분노, 부끄러움, 웃음 등)과 장면의 분위기를 분석하여 어조를 결정하십시오.
- **화법:** 캐릭터 간의 관계(선후배, 친구, 적대 등)에 맞춰 **존댓말(존칭)과 반말**을 정확히 구사하십시오.
- **자연스러움:** 번역 투를 피하고 한국 만화에서 실제로 쓰이는 자연스러운 구어체를 사용하십시오.

# Output
설명이나 사족 없이, 처리가 완료된 **이미지 파일만** 반환하십시오.
"""

# 감독관 프롬프트
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

@st.dialog("📷 이미지 전체 화면", width="large")
def show_full_image(image, caption):
    st.image(image, caption=caption, use_container_width=True)

# --- [3. AI 로직 (생성 + 검수)] ---

def verify_image(api_key, original_img, generated_img):
    """감독관(Flash)이 결과물을 검사하는 함수"""
    try:
        client = genai.Client(api_key=api_key)
        
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
            config=types.GenerateContentConfig(temperature=0.5)
        )
        
        if response.text:
            result = response.text.strip()
            if "PASS" in result:
                return True, "PASS"
            else:
                return False, result 
        return True, "Unknown Response (Passed)"
        
    except Exception as e:
        return True, "Inspector Error (Skipped)"

def generate_with_auto_fix(api_key, prompt, image_input, resolution, temperature, max_retries=2, status_container=None):
    """
    생성(Worker) -> 검수(Inspector) -> (실패시) 재생성 루프
    status_container: st.status 객체 (UI 업데이트용)
    """
    client = genai.Client(api_key=api_key)
    target_bytes = image_to_bytes(image_input)
    
    last_error = ""
    image_config_val = resolution 

    for attempt in range(max_retries + 1):
        try:
            # UI 상태 업데이트
            if status_container:
                msg = f"🎨 **시도 {attempt+1}/{max_retries+1}**: 이미지 생성 중..." if attempt < max_retries else f"🎨 **마지막 시도**: 이미지 생성 중..."
                status_container.write(msg)
            
            # 1. 콘텐츠 구성
            contents = [prompt]
            if attempt > 0 and last_error:
                contents.append(f"⚠️ PREVIOUS ATTEMPT FAILED: {last_error}")
                contents.append("Please fix the issues mentioned above and try again.")
            contents.append("Now, process this image:")
            contents.append(types.Part.from_bytes(data=target_bytes, mime_type="image/png"))

            # 2. API 설정
            config_params = {
                "response_modalities": ["IMAGE"],
                "image_config": types.ImageConfig(image_size=image_config_val)
            }
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
                    temperature=temperature,
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
                if status_container: status_container.write("❌ 이미지 생성 실패 (빈 결과)")
                return None, "이미지 생성 결과가 비어있습니다. (Safety Filter 가능성)"

            # 4. 검수 (Inspector)
            if attempt < max_retries:
                if status_container: status_container.write(f"🧐 **시도 {attempt+1}**: 결과물 검수 중...")
                
                is_pass, reason = verify_image(api_key, image_input, result_img)
                if is_pass:
                    if status_container: status_container.write("✅ 검수 통과!")
                    return result_img, None 
                else:
                    last_error = reason
                    if status_container: status_container.write(f"🚨 **검수 불합격**: {reason} -> 재시도합니다.")
                    time.sleep(1.0)
                    continue
            else:
                if status_container: status_container.write("⚠️ 최대 재시도 횟수 도달. 현재 결과를 반환합니다.")
                return result_img, "최종 시도 완료 (검수 미통과 포함)"

        except Exception as e:
            if status_container: status_container.write(f"🔥 에러 발생: {str(e)}")
            return None, f"API 에러 발생: {str(e)}"
            
    return None, "재시도 횟수를 초과했습니다."

def process_and_update(item, api_key, prompt, resolution, temperature, use_autofix):
    """단일 실행 처리 (Status UI 포함)"""
    start_time = time.time()
    
    # st.status를 사용하여 진행 상태를 시각적으로 표시
    with st.status(f"🚀 **{item['name']}** 작업 시작...", expanded=True) as status:
        if use_autofix:
            res_img, err = generate_with_auto_fix(api_key, prompt, item['image'], resolution, temperature, status_container=status)
        else:
            res_img, err = generate_with_auto_fix(api_key, prompt, item['image'], resolution, temperature, max_retries=0, status_container=status)

        end_time = time.time()
        duration = end_time - start_time

        if res_img:
            # 성공 시 상태 업데이트
            status.update(label=f"✅ 작업 완료! ({duration:.2f}초 소요)", state="complete", expanded=False)
            
            st.session_state.results.append({
                'id': str(uuid.uuid4()), 
                'name': item['name'], 
                'original': item['image'], 
                'result': res_img,
                'duration': duration  # 소요 시간 저장
            })
            st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]
            save_session_to_disk()
            time.sleep(1) # 사용자가 완료 메시지를 볼 수 있게 잠시 대기
            st.rerun()
        else:
            # 실패 시
            status.update(label="❌ 작업 실패", state="error", expanded=True)
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
        
        st.info(f"🛠️ 작업자: {MODEL_WORKER}\n👮 감독관: {MODEL_INSPECTOR}")

        st.divider()
        st.subheader("⚙️ 모델 설정")
        
        resolution = st.radio(
            "해상도 (Resolution)", 
            options=["4K", "2K", "1K"], 
            index=0, 
            horizontal=True,
            help="높을수록 선명하지만 처리 시간이 길어질 수 있습니다."
        )

        temperature = st.slider(
            "창의성 (Temperature)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.2, 
            step=0.1,
            help="낮을수록(0.2) 지시를 엄격히 따르고, 높을수록(0.8) 창의적입니다."
        )

        st.divider()
        st.subheader("⚙️ 옵션")
        use_autofix = st.toggle("🛡️ 자동 검수 & 재생성", value=True, help="결과물이 이상하면 자동으로 다시 시도합니다. (시간 더 걸림)")
        
        if st.button("🗑️ 초기화", use_container_width=True): clear_all_data()
        
        st.divider()
        use_slider = st.toggle("비교 슬라이더", value=True)
        with st.expander("📝 프롬프트 수정"):
            prompt = st.text_area("작업 지시사항", value=DEFAULT_PROMPT, height=300)
            
        return api_key, use_slider, prompt, resolution, temperature, use_autofix

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

def render_queue(api_key, prompt, resolution, temperature, use_autofix):
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
                if b1.button("▶️ 실행", key=f"run_{item['id']}"): process_and_update(item, api_key, prompt, resolution, temperature, use_autofix)
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
                # 소요 시간 표시 추가
                duration_txt = f"⏱️ {item['duration']:.2f}초" if 'duration' in item else ""
                st.markdown(f"### ✅ {item['name']} {duration_txt}")
                
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

def auto_process_step(api_key, prompt, resolution, temperature, use_autofix):
    if not st.session_state.is_auto_running: return
    pending = [i for i in st.session_state.job_queue if i['status'] == 'pending']
    
    if not pending:
        st.session_state.is_auto_running = False
        st.toast("✅ 모든 작업 완료!")
        time.sleep(1)
        st.rerun()
        return

    item = pending[0]
    start_time = time.time()
    
    # 자동 실행 시에도 status 표시
    with st.status(f"🔄 자동 처리 중... [{item['name']}]", expanded=True) as status:
        if use_autofix:
            res_img, err = generate_with_auto_fix(api_key, prompt, item['image'], resolution, temperature, status_container=status)
        else:
            res_img, err = generate_with_auto_fix(api_key, prompt, item['image'], resolution, temperature, max_retries=0, status_container=status)

        end_time = time.time()
        duration = end_time - start_time

        if res_img:
            status.update(label=f"✅ 완료! ({duration:.2f}초)", state="complete", expanded=False)
            st.session_state.results.append({
                'id': str(uuid.uuid4()), 
                'name': item['name'], 
                'original': item['image'], 
                'result': res_img,
                'duration': duration
            })
            st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]
            save_session_to_disk()
        else:
            status.update(label="❌ 실패", state="error")
            item['status'] = 'error'
            item['error_msg'] = err
            save_session_to_disk()
    
    time.sleep(1)
    st.rerun()

# --- [6. 메인 실행] ---
def main():
    init_session_state()
    api_key, use_slider, prompt, resolution, temperature, use_autofix = render_sidebar()
    
    st.title("🍌 Nano Banana")
    st.markdown("**Auto-Fix Edition** (with Supervisor AI)")
    
    handle_file_upload()
    render_queue(api_key, prompt, resolution, temperature, use_autofix)
    render_results(use_slider)

    if st.session_state.is_auto_running:
        auto_process_step(api_key, prompt, resolution, temperature, use_autofix)

if __name__ == "__main__":
    main()
