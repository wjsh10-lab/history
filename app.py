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
MODEL_CHOICES = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-pro"]
HISTORY_LIMIT = 6 # 429 에러 발생 시 유지할 최근 대화 턴 수
RETRY_MAX_ATTEMPTS = 3 # 최대 재시도 횟수

# --- 시스템 프롬프트 ---
SYSTEM_INSTRUCTION = """
당신은 사용자를 미스터리/역사 속으로 안내하는 지식 풍부한 역사 선생님이자 롤플레잉 전문가입니다.
1. **롤플레잉 및 어조**: 사용자가 미스터리/역사에 대해 질문하면, 당신은 마치 그 당시 역사 속으로 들어간 것처럼 롤플레잉을 시작합니다. 재밌고 차분한 어조로, 친절하게 지식을 알려주는 역사 선생님처럼 행동하세요.
2. **정보 수집 및 안내**: 사용자가 물어보는 역사적 사실(사건, 인물 등)에 대해 '무엇이, 언제, 어디서, 어떻게' 일어났는지 자세히 정리하여 수집합니다. 이를 당시 역사에 실제로 존재하는 사람처럼 사용자에게 흥미롭게 안내하세요. 특히, **자세한 년도와 날짜, 그리고 관련 인물에 대한 정보**를 상세히 알려주는 것에 중점을 둡니다.
3. **마무리 및 유도**: 답변 마지막에는 역사/미스터리에 대한 내용을 다시 한번 더 핵심만 정리해주고, 사용자가 그 이야기에 더욱 빠져들 수 있도록 흥미를 유발합니다. 만일 사용자가 다른 역사/미스터리 이야기를 원하면 롤플레잉을 자연스럽게 멈추고, '다른 시대나 미스터리한 이야기에 대해 궁금한 점이 있으신가요?' 와 같이 새로운 질문이 있는지 친절하게 물어보세요.
"""

# --- 함수 정의 ---

def get_api_key():
    """st.secrets에서 API 키를 가져오거나, 사용자에게 임시 입력 UI를 제공합니다."""
    # 1. st.secrets에서 키 확인
    if 'GEMINI_API_KEY' in st.secrets:
        return st.secrets['GEMINI_API_KEY']
    
    # 2. st.secrets에 없을 경우 임시 입력 UI 표시
    st.info("⚠️ **Streamlit Secrets**에 `GEMINI_API_KEY`가 설정되어 있지 않습니다. 아래 입력창에 **임시** API 키를 입력해주세요.")
    # API 입력 UI를 별도의 세션 상태 키로 관리하여 재실행 시 상태 유지
    temp_key = st.text_input("Gemini API Key를 입력하세요:", type="password", key="api_input")
    return temp_key

def initialize_gemini_client(api_key):
    """Gemini 클라이언트를 초기화합니다."""
    try:
        if not api_key:
            return None
        # 클라이언트 객체를 생성하여 반환합니다.
        return genai.Client(api_key=api_key)
    except Exception as e:
        # Streamlit Cloud에서 초기화 오류가 나면 앱이 멈출 수 있으므로 에러만 기록
        print(f"API 클라이언트 초기화 중 오류 발생: {e}")
        return None

def initialize_chat(client, system_instruction, model_name, history):
    """새로운 채팅 세션을 초기화하고 반환합니다."""
    if not client:
        return None
    try:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction
        )
        chat = client.chats.create(
            model=model_name,
            config=config,
            history=history
        )
        return chat
    except Exception as e:
        st.error(f"채팅 세션 초기화 중 오류 발생: {e}")
        return None

def reset_chat_session():
    """대화 세션과 히스토리를 초기화합니다."""
    st.session_state.chat_history = []
    # Chat 객체를 None으로 설정하여 main 로직에서 재초기화를 유도
    st.session_state.chat = None
    st.rerun()

def get_chat_history_for_retry(history, limit):
    """429 에러 발생 시, 최근 N턴만 남기고 히스토리를 잘라냅니다."""
    # 마지막 N개의 Content 객체만 유지
    # 여기서 -1은 마지막 사용자 메시지(재시도할 메시지)를 제외하고 자르기 위함이었으나, 
    # Streamlit 채팅에서는 Chat 객체 자체가 재시도 시 이전 메시지를 포함하므로,
    # 여기서는 안전하게 이전 history의 일부만 남깁니다.
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
    # encoding='utf-8-sig'를 사용하여 BOM을 추가해 엑셀에서 한글 깨짐 방지
    df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    return csv_buffer.getvalue().encode('utf-8-sig')

# --- Streamlit UI 및 메인 로직 ---

st.set_page_config(page_title=CHATBOT_TITLE, layout="wide")
st.title(CHATBOT_TITLE)

# =================================================================
# 1. 세션 상태 초기화 (AttributeError 방지를 위해 최상단에 위치)
# =================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_name" not in st.session_state:
    st.session_state.model_name = DEFAULT_MODEL
if "client" not in st.session_state:
    st.session_state.client = None
if "chat" not in st.session_state:
    st.session_state.chat = None
if "last_api_key" not in st.session_state:
    st.session_state.last_api_key = None
if "log_enabled" not in st.session_state:
    st.session_state.log_enabled = True


# =================================================================
# 2. API 키 설정 및 클라이언트/채팅 객체 초기화 (재시작 로직 포함)
# =================================================================

api_key = get_api_key()

# API 키 변경 또는 클라이언트가 없을 경우 클라이언트 초기화
if api_key and (st.session_state.client is None or st.session_state.last_api_key != api_key):
    st.session_state.client = initialize_gemini_client(api_key)
    st.session_state.last_api_key = api_key
    # 클라이언트가 바뀌었으므로 채팅 객체도 초기화
    st.session_state.chat = initialize_chat(
        st.session_state.client, 
        SYSTEM_INSTRUCTION, 
        st.session_state.model_name, 
        st.session_state.chat_history
    )

# 클라이언트가 없으면 앱 중지
if not st.session_state.client:
    st.error("Gemini API 클라이언트 초기화에 실패했습니다. 유효한 API 키를 입력해주세요.")
    st.stop()


# 모델이 바뀌었거나 Chat 객체가 없을 경우 초기화
if st.session_state.chat is None or st.session_state.chat.model_name != st.session_state.model_name:
    st.session_state.chat = initialize_chat(
        st.session_state.client, 
        SYSTEM_INSTRUCTION, 
        st.session_state.model_name, 
        st.session_state.chat_history
    )


# =================================================================
# 3. 사이드바 설정 (UI)
# =================================================================

with st.sidebar:
    st.header("⚙️ 설정")
    
    # 모델 선택 (세션 상태 model_name에 바인딩)
    st.session_state.model_name = st.selectbox(
        "사용할 기본 모델 선택",
        options=MODEL_CHOICES,
        index=MODEL_CHOICES.index(DEFAULT_MODEL),
        key="model_select",
        on_change=lambda: st.session_state.update(chat=None) # 모델 변경 시 chat 객체 재초기화 유도
    )

    st.markdown("---")
    
    # 대화 초기화 버튼
    if st.button("🗑️ 대화 초기화", help="현재 대화 기록을 모두 지우고 세션을 새로 시작합니다."):
        reset_chat_session()

    st.markdown("---")
    
    # 로그 기록 옵션 및 다운로드
    st.session_state.log_enabled = st.checkbox(
        "💾 CSV 로그 자동 기록", 
        value=st.session_state.log_enabled, # 초기화된 값 사용
        key="log_check", 
        help="모든 대화를 세션 종료 시 자동으로 CSV 파일로 저장합니다."
    )
    
    # 대화 히스토리가 있을 경우 다운로드 버튼 표시
    if st.session_state.chat_history:
        try:
            csv_data = log_conversation_to_csv(st.session_state.chat_history)
            st.download_button(
                label="⬇️ 대화 로그 다운로드 (.csv)",
                data=csv_data,
                file_name=f"history_log_{datetime.date.today()}_{datetime.datetime.now().strftime('%H%M%S')}.csv",
                mime="text/csv",
                help="현재까지의 대화 내용을 CSV 파일로 다운로드합니다."
            )
        except Exception as e:
             st.error(f"로그 다운로드 준비 중 오류 발생: {e}")

    st.markdown("---")

    # 세션 정보 표시
    st.subheader("세션 정보")
    st.info(f"**모델:** `{st.session_state.model_name}`\n\n**대화 턴 수:** `{len(st.session_state.chat_history)}`")


# =================================================================
# 4. 메인 채팅 인터페이스
# =================================================================

# 기존 대화 히스토리 표시
for message in st.session_state.chat_history:
    # 롤 변환: 'model' -> 'assistant'
    role = "assistant" if message.role == "model" else message.role
    with st.chat_message(role):
        st.markdown(message.parts[0].text)

# 사용자 입력 처리
if prompt := st.chat_input("미스터리 또는 역사를 물어보세요..."):
    
    # 사용자 메시지 UI에 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    # 히스토리에 사용자 메시지 추가
    user_content = types.Content(role="user", parts=[types.Part.from_text(prompt)])
    st.session_state.chat_history.append(user_content)

    # 챗봇 응답 생성 및 429 에러 처리 로직
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 429 재시도 로직
        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                # Chat 객체의 send_message를 사용 (재시도 시 히스토리 자동 관리)
                response = st.session_state.chat.send_message(prompt, stream=True)
                for chunk in response:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

                # 성공 시 챗봇 응답을 히스토리에 추가하고 루프 종료
                model_content = types.Content(role="model", parts=[types.Part.from_text(full_response)])
                st.session_state.chat_history.append(model_content)
                break 

            except ResourceExhaustedError:
                if attempt < RETRY_MAX_ATTEMPTS - 1:
                    st.warning(f"⚠️ **429 Rate Limit Exceeded** 발생. 잠시 후 재시도합니다. (시도 {attempt + 1}/{RETRY_MAX_ATTEMPTS})")
                    
                    # 1. Chat History를 최근 6턴만 남기고 잘라냅니다.
                    new_history = get_chat_history_for_retry(st.session_state.chat_history[:-1], HISTORY_LIMIT) # 마지막 사용자 메시지 제외
                    st.session_state.chat_history = new_history
                    
                    # 2. 새로운 (축약된) 히스토리로 Chat 객체 재생성
                    st.session_state.chat = initialize_chat(
                        st.session_state.client, 
                        SYSTEM_INSTRUCTION, 
                        st.session_state.model_name, 
                        st.session_state.chat_history
                    )
                    
                    # 3. 지수 백오프 방식의 대기 (2초, 4초)
                    time.sleep(2 ** (attempt + 1)) 
                    
                    # 4. 재시도 시, 잘려나간 히스토리에 현재 사용자 메시지를 다시 추가
                    st.session_state.chat_history.append(user_content)
                    continue
                else:
                    st.error("❌ **Rate Limit Exceeded**: 할당량 초과. 더 이상 재시도할 수 없습니다. 대화를 초기화합니다.")
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