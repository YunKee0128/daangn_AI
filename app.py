import os
import re
import urllib.parse
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "used_item_price_model.pkl"
DATA_PATH = BASE_DIR / "used_items_cleaned.csv"


st.set_page_config(
    page_title="유학생 중고거래 다국어 도우미",
    page_icon="🥕",
    layout="wide",
)


LANGUAGE_OPTIONS = {
    "English": "English",
    "日本語": "Japanese",
    "中文": "Chinese",
    "Tiếng Việt": "Vietnamese",
    "ไทย": "Thai",
    "한국어": "Korean",
}


def inject_theme_style():
    st.markdown(
        """
        <style>
        :root {
            --karrot: #ff6f0f;
            --karrot-dark: #e85d00;
            --ink: #1f2933;
            --muted: #6b7280;
            --line: #edf0f2;
            --soft: #fff7ed;
            --green: #17a673;
            --red: #e5484d;
            --blue: #2f80ed;
        }

        .stApp {
            background: #fffaf5;
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: rgba(255, 250, 245, 0.92);
            backdrop-filter: blur(10px);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #fff8f1 100%);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] * {
            color: var(--ink);
        }

        [data-testid="stSidebar"] section {
            padding-top: 1.4rem;
        }

        .sidebar-brand {
            background: #ffffff;
            border: 1px solid #ffe2c6;
            border-radius: 18px;
            padding: 18px 16px;
            margin: 4px 0 18px;
            box-shadow: 0 14px 30px rgba(255, 111, 15, 0.10);
        }

        .sidebar-logo {
            width: 42px;
            height: 42px;
            border-radius: 14px;
            background: var(--karrot);
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.45rem;
            font-weight: 900;
            margin-bottom: 12px;
        }

        .sidebar-title {
            font-size: 1.45rem;
            font-weight: 900;
            line-height: 1.22;
            margin-bottom: 8px;
        }

        .sidebar-copy {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .sidebar-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 900;
            margin: 12px 0 8px;
        }

        [data-testid="stSidebar"] [role="radiogroup"] {
            gap: 8px;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            background: #ffffff;
            border: 1px solid #edf0f2;
            border-radius: 14px;
            padding: 12px 13px;
            margin-bottom: 8px;
            box-shadow: 0 8px 18px rgba(31, 41, 51, 0.05);
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            border-color: #ffc391;
            background: #fff7ed;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            border-color: var(--karrot);
            background: #fff1e5;
            box-shadow: 0 10px 20px rgba(255, 111, 15, 0.13);
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
            color: var(--karrot);
            font-weight: 900;
        }

        .sidebar-status {
            display: flex;
            align-items: center;
            gap: 8px;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 12px 13px;
            margin-top: 10px;
            box-shadow: 0 8px 18px rgba(31, 41, 51, 0.05);
            font-size: 0.92rem;
            font-weight: 800;
        }

        .sidebar-status-dot {
            width: 9px;
            height: 9px;
            border-radius: 999px;
            flex: 0 0 auto;
        }

        .sidebar-status.ok .sidebar-status-dot {
            background: var(--green);
        }

        .sidebar-status.warn .sidebar-status-dot {
            background: var(--karrot);
        }

        .sidebar-status.muted .sidebar-status-dot {
            background: #9ca3af;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }

        .app-hero {
            background: linear-gradient(135deg, #fff3e8 0%, #ffffff 58%, #eefaf4 100%);
            border: 1px solid #ffe2c6;
            border-radius: 22px;
            padding: 26px 32px;
            margin-bottom: 22px;
            box-shadow: 0 18px 45px rgba(255, 111, 15, 0.10);
        }

        .hero-kicker {
            color: var(--karrot);
            font-size: 0.92rem;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .hero-title {
            color: var(--ink);
            font-size: clamp(2rem, 3.4vw, 3.25rem);
            font-weight: 900;
            line-height: 1.08;
            margin-bottom: 12px;
        }

        .hero-copy {
            color: #4b5563;
            font-size: 1.05rem;
            line-height: 1.7;
            max-width: 760px;
        }

        .step-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 0 0 18px;
        }

        .step-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 15px 16px;
            box-shadow: 0 10px 24px rgba(31, 41, 51, 0.06);
        }

        .step-num {
            width: 28px;
            height: 28px;
            border-radius: 10px;
            background: #fff1e5;
            color: var(--karrot);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            margin-bottom: 10px;
        }

        .step-title {
            font-size: 1rem;
            font-weight: 900;
            color: var(--ink);
            margin-bottom: 4px;
        }

        .step-copy {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.45;
        }

        .form-heading {
            display: flex;
            align-items: center;
            gap: 9px;
            font-size: 1.08rem;
            font-weight: 900;
            color: var(--ink);
            margin: 2px 0 12px;
        }

        .form-heading span {
            width: 26px;
            height: 26px;
            border-radius: 9px;
            background: var(--karrot);
            color: #ffffff;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
        }

        .form-note {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.55;
            margin: -2px 0 16px;
        }

        .option-row-title {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 900;
            margin: 16px 0 8px;
        }

        div[data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 24px 24px 14px;
            box-shadow: 0 14px 34px rgba(31, 41, 51, 0.08);
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        textarea,
        div[data-baseweb="select"] > div {
            background-color: #fbfbfb !important;
            border-color: #e5e7eb !important;
            color: var(--ink) !important;
            border-radius: 12px !important;
        }

        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stNumberInput"] input:focus,
        textarea:focus {
            border-color: var(--karrot) !important;
            box-shadow: 0 0 0 3px rgba(255, 111, 15, 0.12) !important;
        }

        .stButton > button,
        div[data-testid="stFormSubmitButton"] button {
            background: var(--karrot);
            color: white;
            border: 0;
            border-radius: 12px;
            padding: 0.82rem 1.35rem;
            font-weight: 800;
            box-shadow: 0 10px 22px rgba(255, 111, 15, 0.24);
        }

        .stButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            background: var(--karrot-dark);
            color: white;
            border: 0;
        }

        .info-card {
            min-height: 126px;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 12px 28px rgba(31, 41, 51, 0.07);
            margin-bottom: 14px;
        }

        .info-label {
            color: var(--muted);
            font-size: 0.88rem;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .info-value {
            color: var(--ink);
            font-size: 1.45rem;
            font-weight: 900;
            line-height: 1.25;
            word-break: keep-all;
            overflow-wrap: anywhere;
        }

        .price-panel {
            background: #ffffff;
            border: 1px solid #ffe2c6;
            border-radius: 18px;
            padding: 22px;
            box-shadow: 0 14px 32px rgba(255, 111, 15, 0.10);
            margin: 10px 0 18px;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            padding: 9px 15px;
            border-radius: 999px;
            color: #ffffff;
            font-weight: 900;
            margin-bottom: 10px;
        }

        .small-muted {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.65;
        }

        .ai-panel {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 24px;
            box-shadow: 0 14px 32px rgba(31, 41, 51, 0.07);
        }

        .section-title {
            font-size: 1.45rem;
            font-weight: 900;
            color: var(--ink);
            margin: 26px 0 12px;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 10px 24px rgba(31, 41, 51, 0.06);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        [data-testid="stMetricValue"] {
            color: var(--ink);
        }

        @media (max-width: 780px) {
            .step-strip {
                grid-template-columns: 1fr;
            }

            .app-hero {
                padding: 22px 20px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title, subtitle, kicker="부산외국어대학교 유학생 중고거래 도우미"):
    st.markdown(
        f"""
        <div class="app-hero">
            <div class="hero-kicker">{escape(kicker)}</div>
            <div class="hero-title">{escape(title)}</div>
            <div class="hero-copy">{escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(label, value):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">{escape(str(label))}</div>
            <div class="info-value">{escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)


def load_dotenv_if_exists() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


def make_category(title):
    title = str(title).lower()

    if any(word in title for word in ["아이폰", "갤럭시", "폰", "휴대폰", "노트북", "맥북", "아이패드", "에어팟", "버즈", "컴퓨터", "모니터"]):
        return "전자기기"
    if any(word in title for word in ["책상", "의자", "침대", "서랍", "장롱", "테이블", "소파", "수납"]):
        return "가구"
    if any(word in title for word in ["냉장고", "세탁기", "전자레인지", "청소기", "에어컨", "선풍기", "밥솥"]):
        return "생활가전"
    if any(word in title for word in ["자전거", "운동", "헬스", "축구", "농구", "라켓", "골프"]):
        return "스포츠"
    if any(word in title for word in ["책", "교재", "전공", "토익", "문제집", "소설"]):
        return "도서"
    if any(word in title for word in ["옷", "패딩", "자켓", "신발", "운동화", "가방", "모자"]):
        return "패션"
    return "기타"


def clean_price(price):
    if price is None or pd.isna(price):
        return np.nan

    price = str(price).strip()
    if "무료" in price or "나눔" in price:
        return 0

    numbers = re.sub(r"[^0-9]", "", price)
    if not numbers:
        return np.nan
    return int(numbers)


def format_won(value):
    if value is None or pd.isna(value):
        return "가격 없음"
    return f"{int(value):,}원"


def fix_mojibake(text):
    if text is None:
        return text

    text = str(text)
    broken_markers = ["Ã", "Â", "ì", "í", "ê", "ë", "ï¿½", "�"]
    if not any(marker in text for marker in broken_markers):
        return text

    candidates = [text]
    for source_encoding in ("latin1", "cp1252"):
        try:
            candidates.append(text.encode(source_encoding).decode("utf-8"))
        except UnicodeError:
            pass

    def badness(value):
        return sum(value.count(marker) for marker in broken_markers)

    return min(candidates, key=badness)


def judge_price(actual_price, predicted_price):
    if predicted_price is None or pd.isna(predicted_price):
        return "판단 불가"
    if actual_price < predicted_price * 0.8:
        return "저렴"
    if actual_price > predicted_price * 1.2:
        return "비쌈"
    return "적정"


def get_title_from_url(url):
    try:
        decoded_url = urllib.parse.unquote(url, encoding="utf-8", errors="replace")
        slug = decoded_url.split("/buy-sell/")[-1].strip("/")
        slug = slug.split("?")[0].split("#")[0]
        parts = slug.split("-")
        if len(parts) > 1:
            parts = parts[:-1]
        title = " ".join(parts).strip()
        return fix_mojibake(title or "제목 없음")
    except Exception:
        return "제목 없음"


def clean_description_text(text, title):
    if text is None:
        return "상세설명 없음"

    text = str(text).strip()
    remove_words = [
        "본문 바로가기",
        "당근",
        "중고거래",
        "검색",
        "로그인",
        "회원가입",
        "채팅",
        "관심",
        "조회",
        "신고",
        "끌올",
        "판매중",
        "예약중",
        "거래완료",
    ]

    for word in remove_words:
        text = text.replace(word, " ")
    if title != "제목 없음":
        text = text.replace(title, " ")

    text = re.sub(r"\s+", " ", text).strip()
    return text or "상세설명 없음"


def fetch_html(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        ),
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    response = requests.get(url, headers=headers, timeout=12)
    response.raise_for_status()
    return response.content.decode("utf-8", errors="replace")


def extract_item_info_from_link(url):
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    title = get_title_from_url(url)
    og_title = soup.select_one("meta[property='og:title']")
    if og_title and og_title.get("content"):
        meta_title = og_title["content"].replace(" | 당근", "").replace(" - 당근", "").strip()
        meta_title = fix_mojibake(meta_title)
        if len(meta_title) > len(title):
            title = meta_title

    price = "가격 없음"
    price_pattern = re.compile(r"\d[\d,]*\s*원")
    for line in lines:
        match = price_pattern.search(line)
        if match:
            price = match.group()
            break

    if price == "가격 없음":
        for line in lines:
            if line in ["나눔", "무료", "무료나눔"]:
                price = line
                break

    description = "상세설명 없음"
    for selector in ['meta[property="og:description"]', 'meta[name="description"]']:
        meta_desc = soup.select_one(selector)
        if meta_desc and meta_desc.get("content"):
            description = fix_mojibake(meta_desc["content"].strip())
            break

    if description == "상세설명 없음":
        candidates = []
        for line in lines:
            if len(line) < 8:
                continue
            if any(word in line for word in ["본문 바로가기", "당근", "검색", "로그인", "채팅", "관심", "조회", "신고"]):
                continue
            if title in line or (price != "가격 없음" and price in line):
                continue
            candidates.append(line)
        if candidates:
            description = " ".join(candidates[:5])

    return {
        "제목": fix_mojibake(title),
        "가격": fix_mojibake(price),
        "가격_numeric": clean_price(price),
        "상세설명": fix_mojibake(clean_description_text(description, title)),
        "링크": url,
    }


def predict_price_from_item(item_info, manual_price=None):
    model = load_model()
    title = item_info["제목"]
    category = make_category(title)

    actual_price = manual_price if manual_price is not None else item_info.get("가격_numeric")
    if actual_price is None or pd.isna(actual_price):
        return {
            "제목": title,
            "카테고리": category,
            "실제가격": None,
            "예측적정가격": None,
            "가격판단": "가격 입력 필요",
        }

    actual_price = int(actual_price)
    predicted_price = None
    if model is not None:
        input_df = pd.DataFrame([{
            "제목": title,
            "카테고리": category,
            "제목길이": len(title),
        }])
        predicted_price = max(0, int(model.predict(input_df)[0]))

    return {
        "제목": title,
        "카테고리": category,
        "실제가격": actual_price,
        "예측적정가격": predicted_price,
        "가격판단": judge_price(actual_price, predicted_price),
    }


def build_prompt(item_info, price_result, language):
    return f"""
너는 부산외국어대학교 유학생을 위한 중고거래 도우미야.
아래 상품 정보를 바탕으로 유학생이 이해하기 쉽게 설명해줘.
반드시 {language}로 작성해줘.

[상품 정보]
상품명: {item_info['제목']}
분류: {price_result.get('카테고리', '기타')}
판매가: {format_won(price_result.get('실제가격'))}
예상 시세: {format_won(price_result.get('예측적정가격'))}
가격 상태: {price_result.get('가격판단', '판단 불가')}
판매자 설명: {item_info.get('상세설명', '상세설명 없음')}

답변은 {language}로 자연스럽게 작성하고, 아래 4개 제목도 {language}로 번역해서 써줘.
너무 딱딱한 분석 보고서처럼 쓰지 말고 실제 구매자에게 알려주는 앱 안내처럼 짧고 명확하게 써줘.

### 상품 한눈에 보기
상품이 무엇인지 쉽게 설명해줘.

### 가격 괜찮나요?
판매가가 예상 시세보다 싼지, 적당한지, 비싼지 알려줘. 예상 시세는 앱이 가진 거래 데이터로 계산한 값이라 실제 시세와 다를 수 있다고 짧게 덧붙여줘.

### 사기 전에 확인할 것
구매 전에 확인해야 할 점을 알려줘.

### 판매자에게 보낼 말
판매자에게 보낼 수 있는 짧은 메시지를 만들어줘.
""".strip()


def generate_multilingual_result(item_info, price_result, language):
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI API Key가 없어 구매 도움말을 만들지 못했습니다."

    try:
        from openai import OpenAI
    except Exception:
        return "openai 패키지가 설치되어 있지 않아 구매 도움말을 만들지 못했습니다."

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=get_secret("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for international students buying used goods in Korea."},
            {"role": "user", "content": build_prompt(item_info, price_result, language)},
        ],
    )
    return response.choices[0].message.content


def analyze_used_item_link(url, language="English", manual_price=None, use_ai=True):
    item_info = extract_item_info_from_link(url)
    price_result = predict_price_from_item(item_info, manual_price=manual_price)
    ai_result = None
    if use_ai:
        ai_result = generate_multilingual_result(item_info, price_result, language)
    return {
        "상품정보": item_info,
        "가격판단결과": price_result,
        "AI설명": ai_result,
    }


def render_status_badge(status):
    color = {
        "저렴": "#17a673",
        "적정": "#2f80ed",
        "비쌈": "#e5484d",
        "가격 입력 필요": "#ff6f0f",
        "판단 불가": "#6b7280",
    }.get(status, "#6b7280")
    st.markdown(
        f"""
        <span class="status-pill" style="background:{color};">{escape(str(status))}</span>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_page():
    render_hero(
        "링크만 붙여넣으면 끝",
        "판매가가 괜찮은지 바로 확인하고, 외국어로 보낼 메시지까지 한 번에 준비하세요.",
    )

    st.markdown(
        """
        <div class="step-strip">
            <div class="step-card">
                <div class="step-num">1</div>
                <div class="step-title">링크 붙여넣기</div>
                <div class="step-copy">당근 게시글 주소를 그대로 넣어요.</div>
            </div>
            <div class="step-card">
                <div class="step-num">2</div>
                <div class="step-title">판매가 확인</div>
                <div class="step-copy">자동 인식이 안 되면 직접 입력해요.</div>
            </div>
            <div class="step-card">
                <div class="step-num">3</div>
                <div class="step-title">구매 판단</div>
                <div class="step-copy">예상 시세와 거래 메시지를 확인해요.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("analysis_form"):
        st.markdown(
            """
            <div class="form-heading"><span>1</span>상품 링크</div>
            <div class="form-note">당근마켓 게시글 주소를 복사해서 붙여넣으세요.</div>
            """,
            unsafe_allow_html=True,
        )
        url = st.text_input("상품 링크", placeholder="https://www.daangn.com/kr/buy-sell/...", label_visibility="collapsed")

        st.markdown('<div class="option-row-title">옵션</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns([1.15, 1.2, 1.2, 1.15])
        with col1:
            use_manual_price = st.checkbox("판매가 직접 입력", value=True)
        with col2:
            manual_price = st.number_input("판매가", min_value=0, step=1000, value=0, disabled=not use_manual_price)
        with col3:
            language_label = st.selectbox("언어", list(LANGUAGE_OPTIONS.keys()))
        with col4:
            use_ai = st.checkbox("도움말 받기", value=True)

        submitted = st.form_submit_button("상품 확인하기", type="primary", use_container_width=True)

    if not submitted:
        return

    if not url.strip():
        st.warning("분석할 게시글 링크를 입력해주세요.")
        return

    selected_price = int(manual_price) if use_manual_price else None
    with st.spinner("게시글을 읽고 가격을 분석하는 중입니다..."):
        try:
            result = analyze_used_item_link(
                url=url.strip(),
                language=LANGUAGE_OPTIONS[language_label],
                manual_price=selected_price,
                use_ai=use_ai,
            )
        except Exception as exc:
            st.error(f"분석 중 오류가 발생했습니다: {exc}")
            st.info("당근마켓 페이지 접근이 제한되면 제목/가격을 직접 입력하는 방식으로 테스트해보세요.")
            return

    item_info = result["상품정보"]
    price_result = result["가격판단결과"]

    st.markdown('<div class="section-title">상품 정보</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_info_card("상품명", item_info["제목"])
    with c2:
        render_info_card("판매가", item_info["가격"])
    with c3:
        render_info_card("분류", price_result["카테고리"])
    with c4:
        render_info_card("예상 시세", format_won(price_result["예측적정가격"]))

    st.markdown('<div class="price-panel">', unsafe_allow_html=True)
    st.markdown("#### 가격 체크")
    render_status_badge(price_result["가격판단"])
    st.markdown(
        f"""
        <div class="small-muted">
        판매가 <b>{escape(format_won(price_result['실제가격']))}</b> /
        예상 시세 <b>{escape(format_won(price_result['예측적정가격']))}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("판매자 설명", expanded=True):
        st.write(item_info["상세설명"])

    if result["AI설명"]:
        st.markdown('<div class="section-title">구매 도움말</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(result["AI설명"])


def render_manual_page():
    render_hero(
        "중고거래, 더 쉽게 판단하세요",
        "상품명, 판매가, 판매자 설명을 입력하면 예상 시세와 구매 전 확인할 점을 바로 보여줍니다.",
        kicker="중고거래 가격 체크",
    )

    with st.form("manual_form"):
        title = st.text_input("상품명", placeholder="예: 아이폰 13 128기가 판매합니다")
        col1, col2 = st.columns([1, 1])
        with col1:
            price = st.number_input("판매가", min_value=0, step=1000, value=10000)
        with col2:
            language_label = st.selectbox("설명 언어", list(LANGUAGE_OPTIONS.keys()), key="manual_language")
        description = st.text_area("판매자 설명", placeholder="상품 상태, 구성품, 하자 여부 등을 입력하세요.")
        use_ai = st.checkbox("구매 도움말 만들기", value=True, key="manual_ai")
        submitted = st.form_submit_button("분석하기", type="primary")

    if not submitted:
        return

    if not title.strip():
        st.warning("상품명을 입력해주세요.")
        return

    item_info = {
        "제목": title.strip(),
        "가격": format_won(price),
        "가격_numeric": int(price),
        "상세설명": description.strip() or "상세설명 없음",
        "링크": "",
    }
    price_result = predict_price_from_item(item_info, manual_price=int(price))

    st.markdown('<div class="section-title">가격 분석 결과</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        render_info_card("분류", price_result["카테고리"])
    with col2:
        render_info_card("판매가", format_won(price_result["실제가격"]))
    with col3:
        render_info_card("예상 시세", format_won(price_result["예측적정가격"]))

    st.markdown('<div class="price-panel">', unsafe_allow_html=True)
    st.markdown("#### 가격 체크")
    render_status_badge(price_result["가격판단"])
    st.markdown(
        f"""
        <div class="small-muted">
        예상 시세는 앱이 가진 거래 데이터로 계산한 값입니다. 실제 거래 전에는 상품 상태와 구성품을 함께 확인하세요.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if use_ai:
        with st.spinner("구매 도움말을 만드는 중입니다..."):
            st.markdown('<div class="section-title">구매 도움말</div>', unsafe_allow_html=True)
            ai_text = generate_multilingual_result(item_info, price_result, LANGUAGE_OPTIONS[language_label])
            with st.container(border=True):
                st.markdown(ai_text)


def render_dashboard_page():
    render_hero(
        "수집 데이터 대시보드",
        "부산 지역 중고거래 수집 데이터를 카테고리와 가격 기준으로 살펴볼 수 있습니다.",
        kicker="거래 데이터 보기",
    )
    df = load_data()
    if df.empty:
        st.warning("used_items_cleaned.csv 파일을 찾을 수 없습니다.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("전체 상품 수", f"{len(df):,}")
    col2.metric("평균 가격", format_won(df["가격_numeric"].mean()))
    col3.metric("중앙값", format_won(df["가격_numeric"].median()))
    col4.metric("카테고리 수", df["카테고리"].nunique())

    st.subheader("카테고리별 가격")
    category_summary = (
        df.groupby("카테고리", as_index=False)
        .agg(상품수=("제목", "count"), 평균가격=("가격_numeric", "mean"), 중앙값=("가격_numeric", "median"))
        .sort_values("상품수", ascending=False)
    )
    st.dataframe(category_summary, use_container_width=True)
    st.bar_chart(category_summary.set_index("카테고리")["상품수"])

    st.subheader("상품 검색")
    keyword = st.text_input("검색어", placeholder="아이폰, 책상, 자전거...")
    filtered = df.copy()
    if keyword:
        filtered = filtered[filtered["제목"].astype(str).str.contains(keyword, case=False, na=False)]
    st.dataframe(
        filtered[["수집동네", "제목", "가격", "카테고리", "게시글링크"]].head(200),
        use_container_width=True,
    )


def main():
    load_dotenv_if_exists()
    inject_theme_style()

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">ㄷ</div>
            <div class="sidebar-title">유학생<br>거래 도우미</div>
            <div class="sidebar-copy">가격 확인부터 외국어 메시지까지 한 번에 도와드려요.</div>
        </div>
        <div class="sidebar-label">MENU</div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "메뉴",
        ["직접 입력", "링크 분석", "데이터 대시보드"],
        label_visibility="collapsed",
    )

    model = load_model()
    if model is None:
        st.sidebar.markdown(
            """
            <div class="sidebar-status warn">
                <span class="sidebar-status-dot"></span>
                시세 모델을 찾을 수 없어요
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            """
            <div class="sidebar-status ok">
                <span class="sidebar-status-dot"></span>
                시세 모델 준비 완료
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not get_secret("OPENAI_API_KEY"):
        st.sidebar.markdown(
            """
            <div class="sidebar-status muted">
                <span class="sidebar-status-dot"></span>
                구매 도움말은 키 설정 후 사용 가능
            </div>
            """,
            unsafe_allow_html=True,
        )

    if page == "직접 입력":
        render_manual_page()
    elif page == "링크 분석":
        render_analysis_page()
    else:
        render_dashboard_page()


if __name__ == "__main__":
    main()
