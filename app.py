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
import tempfile # 시스템 임시 폴더 사용

# --- [1. 기본 설정 및 디렉토리 관리] ---
st.set_page_config(page_title="Nano Banana (Cloud)", page_icon="🍌", layout="wide")

try:
    DEFAULT_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    DEFAULT_API_KEY = ""

# 모델 설정
MODEL_WORKER = "gemini-3-pro-image-preview"
MODEL_INSPECTOR = "gemini-3-flash-preview"

# 작업자 프롬프트
DEFAULT_PROMPT = """
# Role
당신은 세계 최고의 "만화 전문 번역 및 식자(Typesetter) AI"입니다. 원본 이미지의 예술적 가치를 완벽하게 보존하면서, 일본어 텍스트를 자연스러운 [한국어]로 변환하여 프로덕션 레벨의 결과물을 완성하십시오.

# 1. 원본 읽기 규칙 (중요: Source Reading Protocol)
- **읽는 순서 (Right-to-Left):** 이 이미지는 일본 만화입니다. 컷의 배치와 말풍선의 순서를 반드시 **오른쪽에서 왼쪽(Right-to-Left)** 방향으로 해석하십시오.
- **문맥 논리:** 오른쪽의 말풍선(질문/원인)을 먼저 해석하고 왼쪽의 말풍선(답변/결과)을 나중에 해석하여, 대화의 인과관계가 뒤바키지 않게 하십시오.

# 2. 시각적 제약 및 원본 보존 (Pixel-Perfect Integrity)
- **[절대 원칙] 원본 훼손 금지:** 텍스트가 있는 말풍선 영역을 제외한 캐릭터, 배경, 펜 선, 스크린톤 등은 **단 1픽셀도 변형하거나 왜곡하지 마십시오.**
- **부분 수정(Inpainting):** 원본 일본어 텍스트만 깨끗이 지우고, 글자 뒤에 가려져 있던 배경(효과선, 배경 패턴 등)을 자연스럽게 복원하십시오.

# 3. 타이포그래피 및 식자 가이드
- **쓰기 방향 (Horizontal):** 읽는 방향과 달리, 번역된 한국어 텍스트는 반드시 **가로쓰기(왼쪽→오른쪽)**로 입력하십시오. **세로쓰기는 절대 금지**입니다.
- **폰트 스타일 매칭:**
  - **대화(Dialogue):** 가독성 좋은 고딕체(Sans-serif) 스타일.
  - **독백/나레이션:** 진지한 느낌의 명조체(Serif) 스타일.
  - **효과음(SFX):** 원본의 거칠거나 굵은 느낌을 살린 붓글씨/디자인 폰트.
- **정렬:** 텍스트는 말풍선 중앙에 배치하고, 테두리에 닿지 않도록 여백을 확보하십시오.

# Output
설명이나 사족 없이, 처리가 완료된 **이미지 파일만** 반환하십시오.
"""

# 감독관 프롬프트 (JSON 출력 강제)
INSPECTOR_PROMPT = """
# Role
You are a QA Supervisor for Korean Manga Localization.

# Task
Compare the [Original Image] and the [Translated Result] and inspect for CRITICAL FAILURES.

# Checklist (Fail Conditions)
1. **Vertical Text:** Is there any Korean text written vertically (Top-to-Bottom)?
2. **Text Overflow:** Is text touching the speech bubble borders or cropped?
3. **Hallucination/Blur:** Is the image blurry, or are faces distorted?
4. **Untranslated:** Is there any original Japanese/English text remaining?
5. **Wrong Language:** Is the output text NOT Korean?

# Output Format (JSON ONLY)
You must return a JSON object. Do not explain textually.
If PASS: {"status": "PASS"}
If FAIL: {"status": "FAIL", "reason": "Brief reason here (e.g. Vertical text detected)"}
"""

# --- [2. 유틸리티 (클라우드 안전 버전)] ---

def save_image_to_temp(image: Image.Image, filename: str) -> str:
    """[수정] 시스템 임시 폴더에 저장 (권한 문제 해결)"""
    # tempfile 모듈을 사용하여 OS가 지정한 임시 폴더(/tmp 등)에 저장
    temp_dir = tempfile.gettempdir()
    safe_name = f"{uuid.uuid4().hex[:8]}_{filename}"
    path = os.path.join(temp_dir, safe_name)
    image.save(path, format="PNG")
    return path

def load_image_from_path(path: str) -> Image.Image:
    """경로에서 이미지를 로드"""
    if path and os.path.exists(path):
        try:
            return Image.open(path)
        except:
            return None
    return None

def init_session_state():
    """[수정] 디스크 로딩(pickle) 제거 -> 메모리 세션만 사용"""
    defaults = {
        'job_queue': [], 
        'results': [],
        'uploader_key': 0, 
        'last_pasted_hash': None, 
        'is_auto_running': False
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

def clear_all_data():
    """[수정] 전체 폴더 삭제 금지 -> 내 세션 데이터만 초기화"""
    st.session_state.job_queue = []
    st.session_state.results = []
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
    """디스크에 저장된 결과물을 ZIP으로 압축"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for item in st.session_state.results:
            img = load_image_from_path(item['result_path']) 
            if img:
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                filename = f"kor_{item['name']}"
                if not filename.lower().endswith('.png'): filename = os.path.splitext(filename)[0] + ".png"
                zip_file.writestr(filename, img_bytes.getvalue())
    return zip_buffer.getvalue()

@st.dialog("📷 이미지 전체 화면", width="large")
def show_full_image(image_path, caption):
    img = load_image_from_path(image_path)
    if img:
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.error("이미지를 찾을 수 없습니다. (세션이 만료되었을 수 있습니다)")

# --- [3. AI 로직 (생성 + 검수)] ---

import json

def verify_image(api_key, original_img, generated_img):
    """JSON 모드로 검수"""
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
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        
        if response.text:
            try:
                data = json.loads(response.text)
                if data.get("status") == "PASS":
                    return True, "PASS"
                else:
                    return False, data.get("reason", "Unknown Failure")
            except json.JSONDecodeError:
                return True, "Inspector JSON Error (Skipped)"
        return True, "No Response (Skipped)"
        
    except Exception as e:
        return True, f"Inspector Error: {e} (Skipped)"

def generate_with_auto_fix(api_key, prompt, image_input, resolution, temperature, max_retries=2, status_container=None):
    """Native API 적용"""
    client = genai.Client(api_key=api_key)
    target_bytes = image_to_bytes(image_input)
    last_error = ""

    for attempt in range(max_retries + 1):
        try:
            if status_container:
                retry_msg = f" (시도 {attempt+1})" if attempt > 0 else ""
                status_container.write(f"🎨 **이미지 생성 중...** {retry_msg} | 해상도: {resolution}")
            
            contents = [prompt]
            if attempt > 0 and last_error:
                contents.append(f"⚠️ PREVIOUS ATTEMPT FAILED: {last_error}")
                contents.append("Please fix the issues mentioned above and try again.")
            contents.append("Now, process this image:")
            contents.append(types.Part.from_bytes(data=target_bytes, mime_type="image/png"))

            config = types.GenerateContentConfig(
                temperature=temperature,
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    image_size=resolution
                ),
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                ]
            )

            response = client.models.generate_content(
                model=MODEL_WORKER,
                contents=contents,
                config=config
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
                return None, "이미지 생성 결과가 비어있습니다."

            if attempt < max_retries:
                if status_container: status_container.write(f"🧐 **품질 검수 중...**")
                
                is_pass, reason = verify_image(api_key, image_input, result_img)
                if is_pass:
                    if status_container: status_container.write("✅ 검수 통과!")
                    return result_img, None 
                else:
                    last_error = reason
                    if status_container: status_container.write(f"🚨 **검수 불합격**: {reason} -> 재시도 중...")
                    time.sleep(1.5)
                    continue
            else:
                if status_container: status_container.write("⚠️ 최대 재시도 횟수 도달. 현재 결과를 반환합니다.")
                return result_img, "최종 시도 완료 (검수 미통과 포함)"

        except Exception as e:
            if status_container: status_container.write(f"🔥 에러 발생: {str(e)}")
            return None, f"API 에러 발생: {str(e)}"
            
    return None, "재시도 횟수를 초과했습니다."

def process_and_update(item, api_key, prompt, resolution, temperature, use_autofix):
    """단일 실행 처리"""
    original_img = load_image_from_path(item['image_path'])
    if not original_img:
        st.error("원본 이미지를 찾을 수 없습니다. (새로고침 시 임시 파일이 삭제되었을 수 있습니다)")
        return

    start_time = time.time()
    
    with st.status(f"🚀 **{item['name']}** 작업 시작...", expanded=True) as status:
        if use_autofix:
            res_img, err = generate_with_auto_fix(api_key, prompt, original_img, resolution, temperature, status_container=status)
        else:
            res_img, err = generate_with_auto_fix(api_key, prompt, original_img, resolution, temperature, max_retries=0, status_container=status)

        end_time = time.time()
        duration = end_time - start_time

        if res_img:
            # [수정] 시스템 임시 폴더에 저장
            res_path = save_image_to_temp(res_img, f"result_{item['name']}")
            
            status.update(label=f"✅ 작업 완료! ({duration:.2f}초)", state="complete", expanded=False)
            
            st.session_state.results.append({
                'id': str(uuid.uuid4()), 
                'name': item['name'], 
                'original_path': item['image_path'], 
                'result_path': res_path,
                'duration': duration
            })
            st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]
            time.sleep(1) 
            st.rerun()
        else:
            status.update(label="❌ 작업 실패", state="error", expanded=True)
            item['status'] = 'error'
            item['error_msg'] = err
            st.rerun()

# --- [4. UI 컴포넌트] ---
from streamlit_paste_button import paste_image_button
from streamlit_image_comparison import image_comparison

def render_sidebar():
    with st.sidebar:
        st.title("🍌 Nano Banana")
        st.caption("Cloud Edition (Safe Mode)")
        api_key = st.text_input("Google API Key", value=DEFAULT_API_KEY, type="password")
        
        st.info(f"🛠️ 작업자: {MODEL_WORKER}\n👮 감독관: {MODEL_INSPECTOR}")

        st.divider()
        st.subheader("⚙️ 모델 설정")
        
        resolution = st.radio(
            "해상도 (Resolution)", 
            options=["4K", "2K", "1K"], 
            index=0, 
            horizontal=True
        )

        temperature = st.slider(
            "창의성 (Temperature)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.2, 
            step=0.1
        )

        st.divider()
        st.subheader("⚙️ 옵션")
        use_autofix = st.toggle("🛡️ 자동 검수 & 재생성", value=True)
        
        # [수정] 클라우드에서는 내 데이터만 지움
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
        with st.spinner("파일 저장 중..."):
            for f in files:
                if f.name.lower().endswith('.zip'):
                    try:
                        with zipfile.ZipFile(f) as z:
                            img_files = [n for n in z.namelist() if n.lower().endswith(('.png','.jpg')) and '__MACOSX' not in n]
                            for fname in img_files:
                                with z.open(fname) as img_f:
                                    img = Image.open(io.BytesIO(img_f.read()))
                                    path = save_image_to_temp(img, os.path.basename(fname))
                                    st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': os.path.basename(fname), 'image_path': path, 'status': 'pending', 'error_msg': None})
                                    new_cnt += 1
                    except: pass
                else:
                    try:
                        img = Image.open(f)
                        path = save_image_to_temp(img, f.name)
                        st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': f.name, 'image_path': path, 'status': 'pending', 'error_msg': None})
                        new_cnt += 1
                    except: pass
            if new_cnt > 0:
                time.sleep(0.5)
                st.session_state.uploader_key += 1
                st.rerun()

    if paste_btn.image_data:
        curr_hash = get_image_hash(paste_btn.image_data)
        if st.session_state.last_pasted_hash != curr_hash:
            path = save_image_to_temp(paste_btn.image_data, f"paste_{int(time.time())}.png")
            st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': f"paste_{int(time.time())}.png", 'image_path': path, 'status': 'pending', 'error_msg': None})
            st.session_state.last_pasted_hash = curr_hash
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
        st.rerun()

    if st.session_state.is_auto_running: st.progress(100, text="🔄 자동 작업 중...")

    for item in st.session_state.job_queue:
        with st.container(border=True):
            col_img, col_info = st.columns([1, 4])
            with col_img:
                img = load_image_from_path(item['image_path'])
                if img:
                    st.image(img, use_container_width=True)
                    if st.button("🔍 확대", key=f"zoom_q_{item['id']}"): show_full_image(item['image_path'], item['name'])
                else:
                    st.error("이미지 유실됨 (새로고침 권장)")

            with col_info:
                st.markdown(f"**📄 {item['name']}**")
                if item['status'] == 'error': st.error(f"❌ {item['error_msg']}")
                elif item['status'] == 'pending': st.info("⏳ 대기 중")
                
                b1, b2, b3 = st.columns([1, 1, 3])
                if b1.button("▶️ 실행", key=f"run_{item['id']}"): process_and_update(item, api_key, prompt, resolution, temperature, use_autofix)
                if b2.button("🗑️ 삭제", key=f"del_{item['id']}"):
                    st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]
                    st.rerun()

def render_results(use_slider):
    if not st.session_state.results: return

    st.divider()
    c1, c2 = st.columns([4, 1])
    c1.subheader(f"🖼️ 완료 ({len(st.session_state.results)}장)")
    
    if c2.button("🗑️ 비우기"):
        st.session_state.results = []
        st.rerun()

    with st.container():
        zip_data = create_zip_file()
        st.download_button("📦 전체 다운로드 (ZIP)", zip_data, "results.zip", "application/zip", use_container_width=True, type="primary")

    st.divider()
    for item in st.session_state.results:
        with st.container(border=True):
            col_img, col_info = st.columns([1, 3])
            
            orig = load_image_from_path(item['original_path'])
            res = load_image_from_path(item['result_path'])

            with col_img:
                if res:
                    st.image(res, use_container_width=True)
                    if st.button("🔍 확대", key=f"zoom_r_{item['id']}"): show_full_image(item['result_path'], item['name'])
            
            with col_info:
                duration_txt = f"⏱️ {item['duration']:.2f}초" if 'duration' in item else ""
                st.markdown(f"### ✅ {item['name']} {duration_txt}")
                
                if use_slider and orig and res:
                    with st.expander("🆚 비교 보기"):
                        if orig.size != res.size: orig = orig.resize(res.size)
                        image_comparison(img1=orig, img2=res, label1="Original", label2="Trans", in_memory=True)
                
                cols = st.columns(3)
                if cols[0].button("🔄 재작업", key=f"re_{item['id']}"):
                    st.session_state.job_queue.append({'id': str(uuid.uuid4()), 'name': item['name'], 'image_path': item['original_path'], 'status': 'pending', 'error_msg': None})
                    st.session_state.results = [x for x in st.session_state.results if x['id'] != item['id']]
                    st.rerun()
                if cols[1].button("🗑️ 삭제", key=f"rm_{item['id']}"):
                    st.session_state.results = [x for x in st.session_state.results if x['id'] != item['id']]
                    st.rerun()
                
                if res:
                    buf = io.BytesIO()
                    res.save(buf, format="PNG")
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
    
    original_img = load_image_from_path(item['image_path'])
    if not original_img:
        item['status'] = 'error'
        item['error_msg'] = "이미지 파일 유실됨"
        st.rerun()
        return

    start_time = time.time()
    
    with st.status(f"🔄 자동 처리 중... [{item['name']}]", expanded=True) as status:
        if use_autofix:
            res_img, err = generate_with_auto_fix(api_key, prompt, original_img, resolution, temperature, status_container=status)
        else:
            res_img, err = generate_with_auto_fix(api_key, prompt, original_img, resolution, temperature, max_retries=0, status_container=status)

        end_time = time.time()
        duration = end_time - start_time

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
            st.session_state.job_queue = [x for x in st.session_state.job_queue if x['id'] != item['id']]
        else:
            status.update(label="❌ 실패", state="error")
            item['status'] = 'error'
            item['error_msg'] = err
    
    time.sleep(1)
    st.rerun()

# --- [6. 메인 실행] ---
def main():
    init_session_state()
    api_key, use_slider, prompt, resolution, temperature, use_autofix = render_sidebar()
    
    st.title("🍌 Nano Banana")
    st.markdown("**Cloud Edition** (Safe Mode & High Res)")
    
    handle_file_upload()
    render_queue(api_key, prompt, resolution, temperature, use_autofix)
    render_results(use_slider)

    if st.session_state.is_auto_running:
        auto_process_step(api_key, prompt, resolution, temperature, use_autofix)

if __name__ == "__main__":
    main()
