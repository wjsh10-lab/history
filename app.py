import streamlit as st
import google.generativeai as genai
from google.generativeai import types
from google.generativeai.errors import ResourceExhaustedError, APIError
import time
import pandas as pd
import io
import datetime

# --- 설정 및 상수 ---
CHATBOT_TITLE = "🕵️ 미스터리/역사 속으로! AI 롤플레잉 챗봇"
DEFAULT_MODEL = "gemini-2.0-flash"
MODEL_CHOICES = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-pro"] # 사용 가능한 모델 목록 (exp 제외)
HISTORY_LIMIT = 6 # 429 에러 발생 시 유지할 최근 대화 턴 수

# --- 시스템 프롬프트 ---
SYSTEM_INSTRUCTION = """
당신은 사용자를 미스터리/역사 속으로 안내하는 지식 풍부한 역사 선생님이자 롤플레잉 전문가입니다.
1. **롤플레잉 및 어조**: 사용자가 미스터리/역사에 대해 질문하면, 당신은 마치 그 당시 역사 속으로 들어간 것처럼 롤플레잉을 시작합니다. 재밌고 차분한 어조로, 친절하게 지식을 알려주는 역사 선생님처럼 행동하세요.
2. **정보 수집 및 안내**: 사용자가 물어보는 역사적 사실(사건, 인물 등)에 대해 '무엇이, 언제, 어디서, 어떻게' 일어났는지 자세히 정리하여 수집합니다. 이를 당시 역사에 실제로 존재하는 사람처럼 사용자에게 흥미롭게 안내하세요. 특히, **자세한 년도와 날짜, 그리고 관련 인물에 대한 정보**를 상세히 알려주는 것에 중점을 둡니다.
3. **마무리 및 유도**: 답변 마지막에는 역사/미스터리에 대한 내용을 다시 한번 더 핵심만 정리해주고, 사용자가 그 이야기에 더욱 빠져들 수 있도록 흥미를 유발합니다. 만일 사용자가 다른 역사/미스터리 이야기를 원하면 롤플레잉을 자연스럽게 멈추고, '다른 시대나 미스터리한 이야기에 대해 궁금한 점이 있으신가요?' 와 같이 새로운 질문이 있는지 친절하게 물어보세요.
"""

# --- API 설정 및 초기화 ---

def get_api_key():
    """st.secrets에서 API 키를 가져오거나, 사용자에게 임시 입력 UI를 제공합니다."""
    # 1. st.secrets에서 키 확인
    if 'GEMINI_API_KEY' in st.secrets:
        return st.secrets['GEMINI_API_KEY']
    
    # 2. st.secrets에 없을 경우 임시 입력 UI 표시
    st.info("⚠️ **Streamlit Secrets**에 `GEMINI_API_KEY`가 설정되어 있지 않습니다. 아래 입력창에 **임시** API 키를 입력해주세요.")
    temp_key = st.text_input("Gemini API Key를 입력하세요:", type="password", key="api_input")
    return temp_key

def initialize_gemini_client(api_key):
    """Gemini 클라이언트를 초기화합니다."""
    try:
        if not api_key:
            return None
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"API 키 초기화 중 오류 발생: {e}")
        return None

def initialize_chat(client, system_instruction, model_name):
    """새로운 채팅 세션을 초기화하고 세션 상태에 저장합니다."""
    try:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction
        )
        chat = client.chats.create(
            model=model_name,
            config=config,
            history=st.session_state.chat_history
        )
        st.session_state.chat = chat
    except Exception as e:
        st.error(f"채팅 세션 초기화 중 오류 발생: {e}")

# --- 대화 히스토리 관리 ---

def reset_chat_session():
    """대화 세션과 히스토리를 초기화합니다."""
    st.session_state.chat_history = []
    if 'client' in st.session_state and st.session_state.client:
        initialize_chat(st.session_state.client, SYSTEM_INSTRUCTION, st.session_state.model_name)
    st.rerun()

def get_chat_history_for_retry(history, limit):
    """429 에러 발생 시, 최근 N턴만 남기고 히스토리를 잘라냅니다."""
    # history는 list of Content 객체
    # 'user'와 'model'이 한 쌍이므로, limit은 짝수로 가정하고 2배를 자릅니다.
    # 안전하게, 마지막 limit개의 Content 객체를 반환
    return history[-limit:]

def log_conversation_to_csv(chat_history):
    """대화 히스토리를 Pandas DataFrame으로 변환하여 CSV 형식의 바이트 스트림을 반환합니다."""
    data = []
    for message in chat_history:
        # Content 객체의 role과 parts[0].text를 추출
        role = "사용자" if message.role == "user" else "챗봇"
        text = message.parts[0].text if message.parts and hasattr(message.parts[0], 'text') else ""
        data.append({"Role": role, "Message": text, "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    df = pd.DataFrame(data)
    
    # CSV 파일로 인코딩 (UTF-8, BOM 포함하여 한글 깨짐 방지)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    return csv_buffer.getvalue().encode('utf-8-sig')

# --- Streamlit UI 및 메인 로직 ---

st.set_page_config(page_title=CHATBOT_TITLE, layout="wide")
st.title(CHATBOT_TITLE)

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 1. 모델 선택
    st.session_state.model_name = st.selectbox(
        "사용할 기본 모델 선택",
        options=MODEL_CHOICES,
        index=MODEL_CHOICES.index(DEFAULT_MODEL),
        key="model_select"
    )

    st.markdown("---")
    
    # 2. 대화 초기화 버튼
    if st.button("🗑️ 대화 초기화", help="현재 대화 기록을 모두 지우고 세션을 새로 시작합니다."):
        reset_chat_session()

    st.markdown("---")
    
    # 3. 로그 기록 옵션 및 다운로드
    st.session_state.log_enabled = st.checkbox("💾 CSV 로그 자동 기록", value=True, key="log_check", help="모든 대화를 세션 종료 시 자동으로 CSV 파일로 저장합니다.")
    
    if st.session_state.chat_history:
        csv_data = log_conversation_to_csv(st.session_state.chat_history)
        st.download_button(
            label="⬇️ 대화 로그 다운로드 (.csv)",
            data=csv_data,
            file_name=f"history_log_{datetime.date.today()}.csv",
            mime="text/csv",
            help="현재까지의 대화 내용을 CSV 파일로 다운로드합니다."
        )

    st.markdown("---")

    # 4. 세션 정보 표시
    st.subheader("세션 정보")
    st.info(f"**모델:** `{st.session_state.model_name}`\n\n**대화 턴 수:** `{len(st.session_state.chat_history)}`")


# --- 메인 앱 로직 ---

# 0. API 키 가져오기 및 클라이언트 초기화
api_key = get_api_key()
if 'client' not in st.session_state or st.session_state.get('last_api_key') != api_key:
    st.session_state.client = initialize_gemini_client(api_key)
    st.session_state.last_api_key = api_key # 키 변경 감지용

if not st.session_state.client:
    st.warning("Gemini API 키를 설정해주세요.")
    st.stop()

# 1. 세션 상태 초기화 (대화 기록 및 Chat 객체)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat" not in st.session_state or st.session_state.chat.model_name != st.session_state.model_name:
    # 모델 변경 감지 또는 Chat 객체가 없을 때 새로 초기화
    initialize_chat(st.session_state.client, SYSTEM_INSTRUCTION, st.session_state.model_name)

# 2. 기존 대화 히스토리 표시
for message in st.session_state.chat_history:
    role = "assistant" if message.role == "model" else message.role
    with st.chat_message(role):
        st.markdown(message.parts[0].text)

# 3. 사용자 입력 처리
if prompt := st.chat_input("미스터리 또는 역사를 물어보세요..."):
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    # 히스토리에 사용자 메시지 추가
    st.session_state.chat_history.append(types.Content(role="user", parts=[types.Part.from_text(prompt)]))

    # 챗봇 응답 생성 및 429 에러 처리 로직
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 429 재시도 로직 (최대 3회 시도)
        for attempt in range(3):
            try:
                # 스트리밍 응답
                response = st.session_state.chat.send_message(prompt, stream=True)
                for chunk in response:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

                # 성공 시 히스토리에 챗봇 응답 추가하고 루프 종료
                st.session_state.chat_history.append(types.Content(role="model", parts=[types.Part.from_text(full_response)]))
                break 

            except ResourceExhaustedError:
                if attempt < 2:
                    st.warning(f"⚠️ **429 Rate Limit Exceeded** 발생. 잠시 후 재시도합니다. (시도 {attempt + 1}/3)")
                    
                    # 최근 6턴만 남기고 히스토리를 잘라내고 재시작
                    new_history = get_chat_history_for_retry(st.session_state.chat_history[:-1], HISTORY_LIMIT) # 마지막 사용자 메시지 제외
                    st.session_state.chat_history = new_history
                    
                    # 새로운 히스토리로 Chat 객체 재생성
                    initialize_chat(st.session_state.client, SYSTEM_INSTRUCTION, st.session_state.model_name)
                    
                    # 지수 백오프 대신 Streamlit 환경을 고려한 고정 대기 시간
                    time.sleep(2 ** (attempt + 1)) 
                    continue
                else:
                    st.error("❌ **Rate Limit Exceeded**: 할당량 초과. 잠시 후 다시 시도하거나, API 키의 할당량을 확인해주세요. 대화를 초기화합니다.")
                    reset_chat_session()
                    break

            except APIError as e:
                st.error(f"❌ **API 오류 발생**: {e}. 대화를 초기화합니다.")
                reset_chat_session()
                break

            except Exception as e:
                st.error(f"❌ **예상치 못한 오류 발생**: {e}. 대화를 초기화합니다.")
                reset_chat_session()
                break