import os
import re
import urllib.parse
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
    page_icon="🛒",
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
        decoded_url = urllib.parse.unquote(url)
        slug = decoded_url.split("/buy-sell/")[-1].strip("/")
        slug = slug.split("?")[0].split("#")[0]
        parts = slug.split("-")
        if len(parts) > 1:
            parts = parts[:-1]
        title = " ".join(parts).strip()
        return title or "제목 없음"
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
    return response.text


def extract_item_info_from_link(url):
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    title = get_title_from_url(url)
    og_title = soup.select_one("meta[property='og:title']")
    if og_title and og_title.get("content"):
        meta_title = og_title["content"].replace(" | 당근", "").replace(" - 당근", "").strip()
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
            description = meta_desc["content"].strip()
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
        "제목": title,
        "가격": price,
        "가격_numeric": clean_price(price),
        "상세설명": clean_description_text(description, title),
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
카테고리: {price_result.get('카테고리', '기타')}
판매 가격: {format_won(price_result.get('실제가격'))}
수집 데이터 기준 참고 가격: {format_won(price_result.get('예측적정가격'))}
가격 판단: {price_result.get('가격판단', '판단 불가')}
상세설명: {item_info.get('상세설명', '상세설명 없음')}

답변 형식:
### 1. Product Summary
상품이 무엇인지 쉽게 설명해줘.

### 2. Price Check
판매 가격이 참고 가격과 비교했을 때 저렴한지, 적정한지, 비싼지 설명해줘. 참고 가격은 수집 데이터 기반 예측값이라 절대적인 시세가 아니라고 자연스럽게 알려줘.

### 3. Things to Check Before Buying
구매 전에 확인해야 할 점을 알려줘.

### 4. Message to Seller
판매자에게 보낼 수 있는 짧은 메시지를 만들어줘.

### 5. Simple Korean Words
게시글에 나오는 중요한 한국어 단어를 {language}로 설명해줘.
""".strip()


def generate_multilingual_result(item_info, price_result, language):
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY가 설정되지 않아 AI 다국어 설명을 생성하지 않았습니다."

    try:
        from openai import OpenAI
    except Exception:
        return "openai 패키지가 설치되어 있지 않아 AI 다국어 설명을 생성하지 않았습니다."

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
        "저렴": "#0f766e",
        "적정": "#2563eb",
        "비쌈": "#dc2626",
        "가격 입력 필요": "#9333ea",
        "판단 불가": "#6b7280",
    }.get(status, "#6b7280")
    st.markdown(
        f"""
        <div style="display:inline-block;padding:8px 14px;border-radius:999px;
        background:{color};color:white;font-weight:700;">{status}</div>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_page():
    st.title("유학생 중고거래 다국어 도우미")
    st.caption("당근마켓 게시글 링크를 분석해 상품 정보, 참고 가격, 다국어 거래 안내를 제공합니다.")

    with st.form("analysis_form"):
        url = st.text_input("당근마켓 게시글 링크", placeholder="https://www.daangn.com/kr/buy-sell/...")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            use_manual_price = st.checkbox("판매 가격 직접 입력", value=True)
        with col2:
            manual_price = st.number_input("판매 가격", min_value=0, step=1000, value=0, disabled=not use_manual_price)
        with col3:
            language_label = st.selectbox("설명 언어", list(LANGUAGE_OPTIONS.keys()))

        use_ai = st.checkbox("AI 다국어 설명 생성", value=True)
        submitted = st.form_submit_button("분석하기", type="primary")

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

    st.subheader("상품 정보")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("상품명", item_info["제목"])
    c2.metric("추출 가격", item_info["가격"])
    c3.metric("카테고리", price_result["카테고리"])
    c4.metric("참고 가격", format_won(price_result["예측적정가격"]))

    st.markdown("#### 가격 판단")
    render_status_badge(price_result["가격판단"])
    st.write(
        f"판매 가격 {format_won(price_result['실제가격'])} / "
        f"수집 데이터 기준 참고 가격 {format_won(price_result['예측적정가격'])}"
    )

    with st.expander("추출된 상세설명", expanded=True):
        st.write(item_info["상세설명"])

    if result["AI설명"]:
        st.subheader("AI 다국어 설명")
        st.markdown(result["AI설명"])


def render_manual_page():
    st.title("직접 입력 가격 분석")
    st.caption("링크 접근이 안 될 때 상품명과 가격을 직접 입력해서 참고 가격을 확인할 수 있습니다.")

    with st.form("manual_form"):
        title = st.text_input("상품명", placeholder="예: 아이폰 13 128기가 판매합니다")
        price = st.number_input("판매 가격", min_value=0, step=1000, value=10000)
        description = st.text_area("상세설명", placeholder="상품 상태, 구성품, 하자 여부 등을 입력하세요.")
        language_label = st.selectbox("설명 언어", list(LANGUAGE_OPTIONS.keys()), key="manual_language")
        use_ai = st.checkbox("AI 다국어 설명 생성", value=True, key="manual_ai")
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

    st.subheader("가격 분석 결과")
    col1, col2, col3 = st.columns(3)
    col1.metric("카테고리", price_result["카테고리"])
    col2.metric("판매 가격", format_won(price_result["실제가격"]))
    col3.metric("참고 가격", format_won(price_result["예측적정가격"]))
    render_status_badge(price_result["가격판단"])

    if use_ai:
        with st.spinner("AI 설명을 생성하는 중입니다..."):
            st.subheader("AI 다국어 설명")
            st.markdown(generate_multilingual_result(item_info, price_result, LANGUAGE_OPTIONS[language_label]))


def render_dashboard_page():
    st.title("수집 데이터 대시보드")
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

    st.sidebar.title("메뉴")
    page = st.sidebar.radio("이동", ["링크 분석", "직접 입력", "데이터 대시보드"])

    model = load_model()
    if model is None:
        st.sidebar.warning("가격 예측 모델 파일이 없습니다.")
    else:
        st.sidebar.success("가격 예측 모델 로드 완료")

    if not get_secret("OPENAI_API_KEY"):
        st.sidebar.info("AI 설명을 사용하려면 OPENAI_API_KEY를 설정하세요.")

    if page == "링크 분석":
        render_analysis_page()
    elif page == "직접 입력":
        render_manual_page()
    else:
        render_dashboard_page()


if __name__ == "__main__":
    main()
