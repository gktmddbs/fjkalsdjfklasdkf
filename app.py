import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image, ImageStat
import io
import os  # [추가] 폴더 생성을 위해 필요
import time
import uuid
import hashlib
from streamlit_paste_button import paste_image_button
from streamlit_image_comparison import image_comparison

# --- 페이지 설정 ---
st.set_page_config(page_title="나노바나나 식질기 (Ultimate v7)", page_icon="🍌", layout="wide")

# --- [세션 상태 초기화] ---
if 'job_queue' not in st.session_state: st.session_state.job_queue = [] 
if 'results' not in st.session_state: st.session_state.results = []
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0
if 'last_pasted_hash' not in st.session_state: st.session_state.last_pasted_hash = None
if 'viewer_mode' not in st.session_state: st.session_state.viewer_mode = False

# --- [유틸리티 함수] ---
def get_image_hash(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return hashlib.md5(img_byte_arr.getvalue()).hexdigest()

def add_to_queue(image, name):
    file_id = str(uuid.uuid4())
    st.session_state.job_queue.append({'id': file_id, 'name': name, 'image': image, 'status': 'pending', 'error_msg': None})

def remove_from_queue(file_id):
    st.session_state.job_queue = [item for item in st.session_state.job_queue if item['id'] != file_id]

def remove_from_results(file_id):
    st.session_state.results = [item for item in st.session_state.results if item['id'] != file_id]

def clear_queue():
    st.session_state.job_queue = []

def resize_image_if_needed(image, max_width):
    if max_width == "원본 유지 (Original)": return image
    target_width = int(max_width)
    if image.width > target_width:
        ratio = target_width / float(image.width)
        return image.resize((target_width, int(float(image.height) * ratio)), Image.Resampling.LANCZOS)
    return image

def analyze_binding_edge(image):
    gray = image.convert('L')
    w, h = gray.size
    crop_w = max(int(w * 0.05), 5)
    left_mean = ImageStat.Stat(gray.crop((0, 0, crop_w, h))).mean[0]
    right_mean = ImageStat.Stat(gray.crop((w - crop_w, 0, w, h))).mean[0]
    return 'left' if left_mean > right_mean else 'right'

# --- [Gemini 처리 함수] ---
def process_single_image(api_key, model_name, image_input, prompt, max_width_setting):
    try:
        processed_input_img = resize_image_if_needed(image_input, max_width_setting)
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        safety_settings = {HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}

        response = model.generate_content([prompt, processed_input_img], safety_settings=safety_settings)
        if not response.candidates: return None, "AI 응답 거부 (필터/과부하)"
        
        result_img = None
        if hasattr(response, 'parts'):
            for part in response.parts:
                if hasattr(part, 'inline_data') and part.inline_data: result_img = Image.open(io.BytesIO(part.inline_data.data))
                elif hasattr(part, 'image') and part.image: result_img = part.image
        
        if result_img: return result_img, None
        return None, "이미지 생성 실패"
    except Exception as e: return None, f"에러: {str(e)}"


# --- [뷰어 모드 UI] ---
def render_viewer_mode():
    """만화 뷰어 모드 (전체화면 CSS 적용)"""
    
    # [핵심] Streamlit의 기본 여백과 헤더를 제거하는 CSS 주입
    st.markdown("""
        <style>
            /* 1. 상단 헤더(햄버거 메뉴 등) 숨기기 */
            header {visibility: hidden;}
            
            /* 2. 하단 푸터 숨기기 */
            footer {visibility: hidden;}
            
            /* 3. 본문 영역 여백 제거 (화면 꽉 채우기) */
            .block-container {
                padding-top: 1rem !important;
                padding-bottom: 0rem !important;
                padding-left: 0rem !important;
                padding-right: 0rem !important;
                max-width: 100% !important;
            }
            
            /* 4. 이미지 간격 제거 및 중앙 정렬 */
            .stImage { margin-bottom: 0px !important; }
            div[data-testid="stImage"] > img {
                display: block;
                margin-left: auto;
                margin-right: auto;
                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.5); /* 그림자 진하게 */
            }
            
            /* 5. 배경색을 검은색에 가까운 회색으로 (몰입감 향상) */
            .stApp {
                background-color: #1E1E1E;
            }
            
            /* 6. 나가기 버튼 등 컨트롤 패널 스타일 */
            .viewer-controls {
                background-color: #333333;
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 10px;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- [상단 컨트롤 패널] ---
    # 컨트롤 패널이 너무 넓으면 방해되므로 중앙에 모음
    with st.container():
        c1, c2 = st.columns([1, 6])
        with c1:
            if st.button("⬅️ 나가기", key="exit_viewer", use_container_width=True):
                st.session_state.viewer_mode = False
                st.rerun()
        with c2:
            with st.expander("⚙️ 뷰어 설정 (화면 조정)", expanded=False):
                vc1, vc2, vc3 = st.columns(3)
                with vc1:
                    view_mode = st.radio("보기 모드", ["스크롤 (웹툰)", "양면 보기 (만화책)"], index=1, horizontal=True)
                with vc2:
                    auto_align = st.toggle("✨ 자동 제본 정렬", value=True)
                    if not auto_align:
                        read_dir = st.radio("방향", ["좌→우", "우←좌 (일본)"], index=1, horizontal=True)
                with vc3:
                    is_cover = st.toggle("첫 장 표지", value=True)
                
                # 이미지 크기 조절 (최대값 대폭 상향)
                img_width = st.slider("화면 확대/축소", 500, 3000, 1200)

    # 결과물 확인
    if not st.session_state.results:
        st.warning("표시할 이미지가 없습니다.")
        return

    results = st.session_state.results
    total = len(results)

    # --- [모드 1: 스크롤] ---
    if view_mode == "스크롤 (웹툰)":
        for idx, item in enumerate(results):
            st.image(item['result'], width=img_width) # 캡션 제거 (몰입 위해)

    # --- [모드 2: 양면 보기] ---
    else:
        idx = 0
        if is_cover and idx < total:
            st.image(results[idx]['result'], width=int(img_width/2))
            idx += 1

        while idx < total:
            curr_res = results[idx]
            current_img = curr_res['result']
            
            # 펼침 페이지 확인
            if current_img.width > current_img.height:
                st.image(current_img, width=img_width)
                idx += 1
            else:
                if idx + 1 < total:
                    next_res = results[idx+1]
                    next_img = next_res['result']
                    
                    if next_img.width <= next_img.height:
                        # 병합 로직
                        max_h = max(current_img.height, next_img.height)
                        def resize_h(img, target_h):
                            return img.resize((int(img.width * (target_h / img.height)), target_h))
                        
                        img_a = resize_h(current_img, max_h)
                        img_b = resize_h(next_img, max_h)
                        
                        # 정렬 로직
                        if auto_align:
                            bind_a = analyze_binding_edge(img_a)
                            left, right = (img_b, img_a) if bind_a == 'left' else (img_a, img_b)
                        else:
                            # 수동
                            if read_dir == "우←좌 (일본)":
                                left, right = img_b, img_a
                            else:
                                left, right = img_a, img_b
                        
                        merged = Image.new('RGB', (left.width + right.width, max_h))
                        merged.paste(left, (0, 0))
                        merged.paste(right, (left.width, 0))
                        
                        st.image(merged, width=img_width)
                        idx += 2
                        continue
                
                # 짝 없을 때
                st.image(current_img, width=int(img_width/2))
                idx += 1
# --- [메인 앱 로직] ---
if st.session_state.viewer_mode: render_viewer_mode()
else:
    st.sidebar.title("🍌 Nano Banana Pro")
    st.sidebar.caption("Ultimate v7 (Local Save)")
    DEFAULT_API_KEY = "AIzaSyBFKyTK2ANjLqY6XX7M4yC_7Xn4WZNucAk"
    api_key = st.sidebar.text_input("Google API Key", value=DEFAULT_API_KEY, type="password")
    model_options = ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-3-pro-image-preview"]
    selected_model = st.sidebar.selectbox("모델 선택", model_options, index=0)
    st.sidebar.divider()
    resolution_options = ["원본 유지 (Original)", "1024", "1280", "1920", "2048"]
    selected_resolution = st.sidebar.selectbox("최대 너비(Width) 제한", resolution_options, index=0)
    use_slider = st.sidebar.toggle("리스트에서 비교 슬라이더 사용", value=True)

    st.sidebar.divider()
    CUSTOM_PROMPT = """
# Role
당신은 세계 최고의 "만화 현지화 전문가"이자 "마스터 식자"입니다.

# Task
제공된 만화 이미지의 일본어/영어를 한국어로 번역하여 자연스럽게 합성(In-painting)하세요.

# Critical Rules (절대 준수)
1. **텍스트 방향 (Orientation):**
   - 일본어의 세로 쓰기를 한국어의 **'가로 쓰기(Horizontal)'**로 반드시 변경하세요.
   - 텍스트가 말풍선 밖으로 삐져나가지 않도록 **줄바꿈(Line break)**을 적절히 사용하세요.

2. **인페인팅 품질 (In-painting Quality):**
   - 원본 글자를 지울 때, 주변 배경(스크린톤, 효과선, 단색 배경)을 분석하여 **위화감 없이 복원**하세요.
   - 글자 뒤에 캐릭터가 있다면, 캐릭터의 선(Lineart)을 뭉개지 말고 살려내야 합니다.

3. **이미지 보존 (Preservation):**
   - **[중요]** 말풍선 내부를 제외한 **나머지 그림(캐릭터, 배경, 프레임)은 1픽셀도 변경하지 마세요.**
   - 이미지의 해상도, 비율, 크기를 원본과 똑같이 유지하세요.

# Style Guide
1. **대사:** 가독성 좋은 고딕 계열(San-serif) 폰트 스타일을 사용하세요. 어조는 생생한 구어체입니다.
2. **효과음:** 원본의 거친 느낌을 살린 그래픽 텍스트로 처리하되, 너무 복잡하면 가독성을 우선시하세요.

# Output
설명 없이, 오직 **편집된 이미지 파일**만 출력하세요."""
    prompt_text = st.sidebar.text_area("시스템 프롬프트", value=CUSTOM_PROMPT, height=200)

    st.title("🍌 나노바나나 식질기 (Ultimate v7)")
    st.markdown("결과물을 **지정된 폴더**에 바로 저장할 수 있습니다.")

    col1, col2 = st.columns([3, 1])
    with col1: uploaded_files = st.file_uploader("이미지 파일 추가", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")
    with col2: st.write("클립보드:"); paste_result = paste_image_button(label="📋 붙여넣기", text_color="#ffffff", background_color="#FF4B4B", hover_background_color="#FF0000")

    if uploaded_files:
        with st.spinner(f"이미지 {len(uploaded_files)}장 로드 중..."):
            for f in uploaded_files:
                try: img = Image.open(f); img.load(); add_to_queue(img, f.name)
                except: st.toast(f"❌ {f.name} 파일 오류")
            time.sleep(0.5); st.session_state.uploader_key += 1; st.rerun()

    if paste_result.image_data is not None:
        try:
            current_img = paste_result.image_data; current_hash = get_image_hash(current_img)
            if st.session_state.last_pasted_hash != current_hash:
                timestamp = int(time.time()); add_to_queue(current_img, f"clipboard_{timestamp}.png")
                st.session_state.last_pasted_hash = current_hash; st.rerun()
        except: pass

    st.divider()
    col_q_header, col_q_btn = st.columns([4, 1])
    with col_q_header: st.subheader(f"📂 작업 대기열 ({len(st.session_state.job_queue)}장)")
    with col_q_btn:
        if len(st.session_state.job_queue) > 0:
            if st.button("🗑️ 대기열 비우기"): clear_queue(); st.rerun()

    if st.session_state.job_queue:
        with st.expander("목록 관리", expanded=True):
            for item in st.session_state.job_queue:
                c1, c2, c3 = st.columns([1, 3, 2])
                with c1: st.image(item['image'], width=100)
                with c2:
                    st.write(f"**{item['name']}**")
                    if item['status'] == 'error': st.error(f"실패: {item['error_msg']}")
                    elif item['status'] == 'pending': st.caption("대기 중...")
                with c3:
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        if item['status'] == 'error':
                            if st.button("🔄 재시도", key=f"retry_{item['id']}"):
                                with st.spinner("재시도 중..."):
                                    res_img, err = process_single_image(api_key, selected_model, item['image'], prompt_text, selected_resolution)
                                    if res_img:
                                        res_id = str(uuid.uuid4()); st.session_state.results.append({'id': res_id, 'name': item['name'], 'original': item['image'], 'result': res_img})
                                        remove_from_queue(item['id']); st.rerun()
                                    else: item['error_msg'] = err; st.rerun()
                    with b_col2:
                        if st.button("❌ 삭제", key=f"del_q_{item['id']}"): remove_from_queue(item['id']); st.rerun()
                st.divider()
        
        pending_items = [i for i in st.session_state.job_queue if i['status'] == 'pending']
        if pending_items:
            if st.button(f"🚀 나머지 {len(pending_items)}장 일괄 시작", type="primary"):
                if not api_key: st.error("API 키 필요")
                else:
                    progress = st.progress(0); status = st.empty(); total = len(pending_items)
                    for idx, item in enumerate(pending_items):
                        status.text(f"처리 중 [{idx+1}/{total}]: {item['name']}")
                        res_img, err = process_single_image(api_key, selected_model, item['image'], prompt_text, selected_resolution)
                        if res_img:
                            res_id = str(uuid.uuid4()); st.session_state.results.append({'id': res_id, 'name': item['name'], 'original': item['image'], 'result': res_img})
                            remove_from_queue(item['id'])
                        else: item['status'] = 'error'; item['error_msg'] = err
                        progress.progress((idx+1)/total); time.sleep(1)
                    status.success("완료!"); st.rerun()

    if st.session_state.results:
        st.divider()
        col_r_header, col_r_btn, col_viewer_btn = st.columns([3, 1, 1])
        with col_r_header: st.subheader(f"🖼️ 결과 ({len(st.session_state.results)}장)")
        with col_r_btn:
            if st.button("🗑️ 결과 비우기"): st.session_state.results = []; st.rerun()
        with col_viewer_btn:
            if st.button("📖 뷰어 모드", type="primary"): st.session_state.viewer_mode = True; st.rerun()
        
        # --- [NEW] 폴더 저장 UI ---
        st.markdown("### 💾 저장 옵션")
        save_c1, save_c2 = st.columns([3, 1])
        with save_c1:
            # 기본값으로 'Nanobanana_Result' 등을 넣어줌
            target_folder = st.text_input("저장할 폴더 이름 (현재 위치에 생성됨)", value="나노바나나_결과물")
        with save_c2:
            if st.button("📂 폴더에 일괄 저장", type="primary", use_container_width=True):
                if not target_folder:
                    st.error("폴더 이름을 입력하세요.")
                else:
                    try:
                        # 폴더 생성 (이미 있으면 무시)
                        os.makedirs(target_folder, exist_ok=True)
                        save_count = 0
                        for item in st.session_state.results:
                            # 파일명 충돌 방지를 위해 edited_ 접두어 붙임
                            # 확장자는 무조건 png로 저장 (가장 안전)
                            safe_name = f"edited_{item['name']}"
                            if not safe_name.lower().endswith('.png'):
                                safe_name += ".png"
                                
                            save_path = os.path.join(target_folder, safe_name)
                            item['result'].save(save_path, format="PNG")
                            save_count += 1
                        
                        st.success(f"✅ 저장 완료! \n\n경로: `{os.path.abspath(target_folder)}` \n\n총 {save_count}장이 저장되었습니다.")
                    except Exception as e:
                        st.error(f"저장 중 오류 발생: {e}")
        st.divider()

        # 리스트 표시
        r_cols = st.columns(2)
        for idx, item in enumerate(st.session_state.results):
            col = r_cols[idx % 2]
            with col:
                st.markdown(f"**{item['name']}**")
                if use_slider:
                    orig_show = item['original']; res_show = item['result']
                    if orig_show.size != res_show.size: orig_show = orig_show.resize(res_show.size)
                    image_comparison(img1=orig_show, img2=res_show, label1="Original", label2="Trans", width=400, in_memory=True)
                else: st.image(item['result'], caption="식질 완료", use_container_width=True)
                
                # 개별 삭제 버튼만 남김 (저장은 위에서 폴더로 하니까)
                if st.button("❌ 삭제", key=f"del_res_{item['id']}"): remove_from_results(item['id']); st.rerun()