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


CATEGORY_OPTIONS = ["전자기기", "가구", "생활가전", "스포츠", "도서", "패션", "기타"]

CATEGORY_LABELS = {
    "전자기기": "전자기기",
    "가구": "가구",
    "생활가전": "생활가전",
    "스포츠": "스포츠",
    "도서": "도서",
    "패션": "패션",
    "기타": "기타",
}

CATEGORY_EXAMPLES = {
    "전자기기": ["아이폰", "노트북", "에어팟", "모니터"],
    "가구": ["책상", "의자", "침대", "수납장"],
    "생활가전": ["냉장고", "전자레인지", "청소기", "선풍기"],
    "스포츠": ["자전거", "축구화", "라켓", "운동기구"],
    "도서": ["전공책", "토익책", "문제집", "소설"],
    "패션": ["패딩", "신발", "가방", "모자"],
    "기타": ["생활용품", "문구", "굿즈", "기타 물건"],
}

CATEGORY_PLACEHOLDERS = {
    "전자기기": "예: 아이폰 13 128기가 판매합니다",
    "가구": "예: 원목 책상 판매합니다",
    "생활가전": "예: 전자레인지 깨끗하게 사용했습니다",
    "스포츠": "예: 자전거 상태 좋아요",
    "도서": "예: 토익 문제집 새 책입니다",
    "패션": "예: 나이키 운동화 판매합니다",
    "기타": "예: 생활용품 일괄 판매합니다",
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
            --soft: #fff4ea;
            --green: #17a673;
            --red: #e5484d;
            --blue: #2f80ed;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 111, 15, 0.08), transparent 28rem),
                linear-gradient(180deg, #ffffff 0%, #fffaf6 100%);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.88);
            backdrop-filter: blur(10px);
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
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
            box-shadow: 0 16px 36px rgba(31, 41, 51, 0.07);
        }

        .sidebar-logo {
            width: 44px;
            height: 44px;
            border-radius: 15px;
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

        .sidebar-actions {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin: 0 0 16px;
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            background: #ffffff;
            color: var(--ink);
            border: 1px solid #edf0f2;
            box-shadow: 0 8px 18px rgba(31, 41, 51, 0.05);
            padding: 0.7rem 0.75rem;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: #fff7ed;
            color: var(--karrot);
            border-color: #ffc391;
        }

        [data-testid="stSidebar"] [role="radiogroup"] {
            gap: 8px;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            position: relative;
            background: #ffffff;
            border: 1px solid #edf0f2;
            border-radius: 14px;
            padding: 13px 14px;
            margin-bottom: 8px;
            box-shadow: 0 8px 18px rgba(31, 41, 51, 0.05);
            cursor: pointer;
            transition: all 0.16s ease;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            border-color: #ffc391;
            background: #fff7ed;
            transform: translateY(-1px);
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            border-color: var(--karrot);
            background: linear-gradient(180deg, #fff7ef 0%, #ffffff 100%);
            box-shadow: 0 10px 20px rgba(255, 111, 15, 0.13);
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
            color: var(--karrot);
            font-weight: 900;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
            display: none;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked)::before {
            content: "";
            position: absolute;
            left: 0;
            top: 12px;
            bottom: 12px;
            width: 4px;
            border-radius: 999px;
            background: var(--karrot);
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
            max-width: 1160px;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }

        .app-hero {
            background: linear-gradient(135deg, #ffffff 0%, #fff7ef 48%, #ffffff 100%);
            border: 1px solid #ffe2c6;
            border-radius: 22px;
            padding: 26px 32px;
            margin-bottom: 22px;
            box-shadow: 0 18px 44px rgba(31, 41, 51, 0.07);
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

        .example-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 8px 0 20px;
        }

        .example-chip {
            background: #fff7ed;
            border: 1px solid #ffe2c6;
            color: #9a4a05;
            border-radius: 999px;
            padding: 7px 11px;
            font-size: 0.86rem;
            font-weight: 800;
        }

        .input-section {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 22px 24px 12px;
            margin-top: 14px;
            box-shadow: 0 14px 34px rgba(31, 41, 51, 0.08);
        }

        div[data-testid="stPills"] button {
            border-radius: 14px !important;
            padding: 12px 16px !important;
            border: 1px solid #edf0f2 !important;
            background: #ffffff !important;
            box-shadow: 0 8px 18px rgba(31, 41, 51, 0.05);
            font-weight: 900 !important;
        }

        div[data-testid="stPills"] button[aria-pressed="true"] {
            border-color: var(--karrot) !important;
            background: #fff1e5 !important;
            color: var(--karrot) !important;
            box-shadow: 0 10px 20px rgba(255, 111, 15, 0.13);
        }

        div[data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 24px 24px 14px;
            box-shadow: 0 18px 46px rgba(31, 41, 51, 0.08);
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        textarea,
        div[data-baseweb="select"] > div {
            background-color: #fbfbfb !important;
            border-color: #e5e7eb !important;
            color: var(--ink) !important;
            border-radius: 14px !important;
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
            border-radius: 14px;
            padding: 0.82rem 1.35rem;
            font-weight: 800;
            box-shadow: 0 10px 22px rgba(255, 111, 15, 0.24);
            transition: all 0.16s ease;
        }

        .stButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            background: var(--karrot-dark);
            color: white;
            border: 0;
            transform: translateY(-1px);
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

        .result-summary {
            background: linear-gradient(135deg, #ffffff 0%, #fff7ef 100%);
            border: 1px solid #ffe2c6;
            border-radius: 22px;
            padding: 24px;
            box-shadow: 0 18px 42px rgba(255, 111, 15, 0.12);
            margin: 24px 0 18px;
        }

        .result-kicker {
            color: var(--karrot);
            font-size: 0.86rem;
            font-weight: 900;
            margin-bottom: 8px;
        }

        .result-badge {
            display: inline-flex;
            align-items: center;
            padding: 8px 12px;
            border-radius: 999px;
            color: #ffffff;
            font-size: 0.9rem;
            font-weight: 900;
            margin-bottom: 12px;
        }

        .result-title {
            color: var(--ink);
            font-size: clamp(1.5rem, 2.6vw, 2.25rem);
            font-weight: 900;
            line-height: 1.22;
            margin-bottom: 8px;
        }

        .result-copy {
            color: #4b5563;
            font-size: 1rem;
            line-height: 1.65;
        }

        .result-numbers {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin-top: 18px;
        }

        .result-number {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 16px;
        }

        .result-number-label {
            color: var(--muted);
            font-size: 0.84rem;
            font-weight: 900;
            margin-bottom: 6px;
        }

        .result-number-value {
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 900;
        }

        .notice-card {
            background: #ffffff;
            border: 1px solid #ffe2c6;
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 12px 28px rgba(31, 41, 51, 0.07);
            margin: 14px 0;
        }

        .notice-title {
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 900;
            margin-bottom: 6px;
        }

        .notice-copy {
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.6;
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

        .dashboard-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 20px;
            box-shadow: 0 12px 28px rgba(31, 41, 51, 0.07);
            margin-bottom: 14px;
        }

        .category-stat {
            background: #ffffff;
            border: 1px solid #edf0f2;
            border-radius: 16px;
            padding: 18px;
            min-height: 128px;
            box-shadow: 0 10px 24px rgba(31, 41, 51, 0.06);
        }

        .category-name {
            font-size: 1.06rem;
            font-weight: 900;
            color: var(--ink);
            margin-bottom: 8px;
        }

        .category-price {
            font-size: 1.35rem;
            font-weight: 900;
            color: var(--karrot);
            margin-bottom: 6px;
        }

        .category-count {
            color: var(--muted);
            font-size: 0.9rem;
            font-weight: 700;
        }

        .item-row {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 10px;
            box-shadow: 0 8px 20px rgba(31, 41, 51, 0.05);
        }

        .item-title {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 900;
            margin-bottom: 8px;
            overflow-wrap: anywhere;
        }

        .item-meta {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.5;
        }

        .item-price {
            color: var(--karrot);
            font-weight: 900;
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

            .result-numbers {
                grid-template-columns: 1fr;
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


def get_price_feedback(status):
    messages = {
        "저렴": (
            "좋은 거래일 가능성이 높아요",
            "판매가가 예상 시세보다 낮은 편입니다. 다만 너무 저렴한 상품은 상태, 구성품, 직거래 장소를 한 번 더 확인하세요.",
        ),
        "적정": (
            "무난한 가격대로 보여요",
            "판매가가 예상 시세와 크게 차이 나지 않습니다. 상품 상태와 구성품이 설명과 맞는지 확인하면 좋아요.",
        ),
        "비쌈": (
            "가격이 높은 편이에요",
            "판매가가 예상 시세보다 높게 나왔습니다. 같은 상품의 다른 매물과 비교하거나 가격 조정을 요청해보세요.",
        ),
        "가격 입력 필요": (
            "판매가를 입력하면 판단할 수 있어요",
            "게시글에서 가격을 읽지 못했습니다. 판매가를 직접 입력하면 예상 시세와 비교해드릴게요.",
        ),
        "판단 불가": (
            "지금은 판단하기 어려워요",
            "모델이나 가격 정보가 부족합니다. 상품명과 가격을 더 구체적으로 입력해보세요.",
        ),
    }
    return messages.get(status, messages["판단 불가"])


def render_price_summary(price_result):
    status = price_result.get("가격판단", "판단 불가")
    title, copy = get_price_feedback(status)
    color = {
        "저렴": "#17a673",
        "적정": "#2f80ed",
        "비쌈": "#e5484d",
        "가격 입력 필요": "#ff6f0f",
        "판단 불가": "#6b7280",
    }.get(status, "#6b7280")
    st.markdown(
        f"""
        <div class="result-summary">
            <div class="result-kicker">가격 체크 결과</div>
            <div class="result-badge" style="background:{color};">{escape(str(status))}</div>
            <div class="result-title">{escape(title)}</div>
            <div class="result-copy">{escape(copy)}</div>
            <div class="result-numbers">
                <div class="result-number">
                    <div class="result-number-label">판매가</div>
                    <div class="result-number-value">{escape(format_won(price_result.get("실제가격")))}</div>
                </div>
                <div class="result-number">
                    <div class="result-number-label">예상 시세</div>
                    <div class="result-number-value">{escape(format_won(price_result.get("예측적정가격")))}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_notice(title, copy):
    st.markdown(
        f"""
        <div class="notice-card">
            <div class="notice-title">{escape(title)}</div>
            <div class="notice-copy">{escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def reset_inputs():
    keys_to_clear = [
        "link_url",
        "link_category",
        "link_manual_price_enabled",
        "link_manual_price",
        "link_language",
        "link_use_ai",
        "manual_category",
        "manual_title",
        "manual_price",
        "manual_language",
        "manual_description",
        "manual_ai",
        "dashboard_category",
        "dashboard_keyword",
        "dashboard_sort",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


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
    category = item_info.get("카테고리") or make_category(title)

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
단, 실제 판매자에게 복사해서 보낼 한국어 채팅 문장만 한국어로 작성해줘.
너무 딱딱한 분석 보고서처럼 쓰지 말고 실제 구매자에게 알려주는 앱 안내처럼 짧고 명확하게 써줘.

### 상품 한눈에 보기
상품이 무엇인지 쉽게 설명해줘.

### 가격 괜찮나요?
판매가가 예상 시세보다 싼지, 적당한지, 비싼지 알려줘. 예상 시세는 앱이 가진 거래 데이터로 계산한 값이라 실제 시세와 다를 수 있다고 짧게 덧붙여줘.

### 사기 전에 확인할 것
구매 전에 확인해야 할 점을 알려줘.

### 판매자에게 보낼 말
판매자에게 보낼 수 있는 짧은 메시지를 먼저 {language}로 만들어줘.
그리고 바로 아래에 "{language}로 번역된 'Send in Korean' 의미의 작은 제목"을 붙이고, 실제 당근 채팅에 복사해서 보낼 수 있는 자연스러운 한국어 메시지도 반드시 함께 써줘.
한국어 메시지는 존댓말로, 너무 길지 않게 1~3문장으로 작성해줘.
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


def analyze_used_item_link(url, language="English", manual_price=None, use_ai=True, category=None):
    item_info = extract_item_info_from_link(url)
    if category:
        item_info["카테고리"] = category
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
        url = st.text_input(
            "상품 링크",
            placeholder="https://www.daangn.com/kr/buy-sell/...",
            label_visibility="collapsed",
            key="link_url",
        )

        st.markdown(
            """
            <div class="form-heading"><span>2</span>상품 종류</div>
            <div class="form-note">링크만으로 종류를 알기 어려울 때를 위해 먼저 골라주세요.</div>
            """,
            unsafe_allow_html=True,
        )
        selected_category = st.pills(
            "상품 종류",
            CATEGORY_OPTIONS,
            default="전자기기",
            format_func=lambda value: CATEGORY_LABELS[value],
            key="link_category",
            label_visibility="collapsed",
            width="stretch",
        )
        selected_category = selected_category or "기타"

        st.markdown('<div class="option-row-title">확인 옵션</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns([1.15, 1.2, 1.2, 1.15])
        with col1:
            use_manual_price = st.checkbox("판매가 직접 입력", value=True, key="link_manual_price_enabled")
        with col2:
            manual_price = st.number_input("판매가", min_value=0, step=1000, value=0, disabled=not use_manual_price, key="link_manual_price")
        with col3:
            language_label = st.selectbox("언어", list(LANGUAGE_OPTIONS.keys()), key="link_language")
        with col4:
            use_ai = st.checkbox("도움말 받기", value=True, key="link_use_ai")

        submitted = st.form_submit_button("상품 확인하기", type="primary", use_container_width=True)

    if not submitted:
        return

    if not url.strip():
        render_notice("상품 링크를 넣어주세요", "당근마켓 게시글 주소를 복사해서 붙여넣으면 바로 확인할 수 있어요.")
        return

    selected_price = int(manual_price) if use_manual_price else None
    with st.spinner("게시글을 읽고 가격을 분석하는 중입니다..."):
        try:
            result = analyze_used_item_link(
                url=url.strip(),
                language=LANGUAGE_OPTIONS[language_label],
                manual_price=selected_price,
                use_ai=use_ai,
                category=selected_category,
            )
        except Exception as exc:
            render_notice(
                "링크를 읽지 못했어요",
                "게시글 접근이 제한되었거나 주소가 올바르지 않을 수 있어요. 같은 상품을 직접 입력 메뉴에서 확인해보세요.",
            )
            st.caption(f"오류 정보: {exc}")
            return

    item_info = result["상품정보"]
    price_result = result["가격판단결과"]

    render_price_summary(price_result)

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

    with st.expander("판매자 설명", expanded=True):
        st.write(item_info["상세설명"])

    if result["AI설명"]:
        st.markdown('<div class="section-title">구매 도움말</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(result["AI설명"])


def render_manual_page():
    render_hero(
        "무슨 물건인지 먼저 골라주세요",
        "상품 종류를 고르면 입력 예시가 바뀌고, 더 자연스럽게 가격을 확인할 수 있어요.",
        kicker="중고거래 시세 확인",
    )

    with st.form("manual_form"):
        st.markdown(
            """
            <div class="form-heading"><span>1</span>상품 종류</div>
            <div class="form-note">가장 가까운 종류를 먼저 선택하세요. 정확히 몰라도 괜찮아요.</div>
            """,
            unsafe_allow_html=True,
        )
        selected_category = st.pills(
            "상품 종류",
            CATEGORY_OPTIONS,
            default="전자기기",
            format_func=lambda value: CATEGORY_LABELS[value],
            key="manual_category",
            label_visibility="collapsed",
            width="stretch",
        )
        selected_category = selected_category or "기타"
        examples = CATEGORY_EXAMPLES.get(selected_category, CATEGORY_EXAMPLES["기타"])
        st.markdown(
            '<div class="example-chips">'
            + "".join(f'<span class="example-chip">{escape(example)}</span>' for example in examples)
            + "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="form-heading"><span>2</span>상품 정보</div>
            <div class="form-note">제목과 판매가만 넣어도 확인할 수 있어요. 설명을 넣으면 구매 도움말이 더 좋아집니다.</div>
            """,
            unsafe_allow_html=True,
        )
        title = st.text_input(
            "상품명",
            placeholder=CATEGORY_PLACEHOLDERS.get(selected_category, CATEGORY_PLACEHOLDERS["기타"]),
            key="manual_title",
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            price = st.number_input("판매가", min_value=0, step=1000, value=10000, key="manual_price")
        with col2:
            language_label = st.selectbox("도움말 언어", list(LANGUAGE_OPTIONS.keys()), key="manual_language")
        description = st.text_area(
            "판매자 설명",
            placeholder="예: 사용 기간, 구성품, 하자 여부, 직거래 위치 등을 적어주세요.",
            key="manual_description",
        )

        st.markdown('<div class="option-row-title">추가 옵션</div>', unsafe_allow_html=True)
        use_ai = st.checkbox("구매 도움말 받기", value=True, key="manual_ai")
        submitted = st.form_submit_button("가격 확인하기", type="primary", use_container_width=True)

    if not submitted:
        return

    if not title.strip():
        render_notice("상품명을 입력해주세요", "예: 아이폰 13, 책상, 자전거처럼 상품을 알아볼 수 있게 적으면 더 정확해져요.")
        return

    item_info = {
        "제목": title.strip(),
        "카테고리": selected_category,
        "가격": format_won(price),
        "가격_numeric": int(price),
        "상세설명": description.strip() or "상세설명 없음",
        "링크": "",
    }
    price_result = predict_price_from_item(item_info, manual_price=int(price))

    render_price_summary(price_result)

    st.markdown('<div class="section-title">가격 분석 결과</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        render_info_card("분류", price_result["카테고리"])
    with col2:
        render_info_card("판매가", format_won(price_result["실제가격"]))
    with col3:
        render_info_card("예상 시세", format_won(price_result["예측적정가격"]))

    st.markdown(
        """
        <div class="small-muted">
        예상 시세는 앱이 가진 거래 데이터로 계산한 값입니다. 실제 거래 전에는 상품 상태와 구성품을 함께 확인하세요.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if use_ai:
        with st.spinner("구매 도움말을 만드는 중입니다..."):
            st.markdown('<div class="section-title">구매 도움말</div>', unsafe_allow_html=True)
            ai_text = generate_multilingual_result(item_info, price_result, LANGUAGE_OPTIONS[language_label])
            with st.container(border=True):
                st.markdown(ai_text)


def render_dashboard_page():
    render_hero(
        "부산 중고거래 시세 보기",
        "수집한 상품 데이터를 카테고리별로 훑어보고, 궁금한 물건을 바로 검색해보세요.",
        kicker="거래 데이터 둘러보기",
    )
    df = load_data()
    if df.empty:
        st.warning("used_items_cleaned.csv 파일을 찾을 수 없습니다.")
        return

    df = df.copy()
    df["가격_numeric"] = pd.to_numeric(df["가격_numeric"], errors="coerce")
    df = df.dropna(subset=["가격_numeric"])

    st.markdown('<div class="section-title">한눈에 보기</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_info_card("등록 상품", f"{len(df):,}개")
    with col2:
        render_info_card("평균 시세", format_won(df["가격_numeric"].mean()))
    with col3:
        render_info_card("중간 가격", format_won(df["가격_numeric"].median()))
    with col4:
        render_info_card("상품 종류", f"{df['카테고리'].nunique()}개")

    category_summary = (
        df.groupby("카테고리", as_index=False)
        .agg(상품수=("제목", "count"), 평균가격=("가격_numeric", "mean"), 중앙값=("가격_numeric", "median"))
        .sort_values("상품수", ascending=False)
    )

    st.markdown('<div class="section-title">카테고리별 시세</div>', unsafe_allow_html=True)
    category_cols = st.columns(3)
    for index, row in category_summary.head(6).reset_index(drop=True).iterrows():
        with category_cols[index % 3]:
            st.markdown(
                f"""
                <div class="category-stat">
                    <div class="category-name">{escape(str(row['카테고리']))}</div>
                    <div class="category-price">{escape(format_won(row['중앙값']))}</div>
                    <div class="category-count">{int(row['상품수']):,}개 상품 · 평균 {escape(format_won(row['평균가격']))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">상품 찾아보기</div>', unsafe_allow_html=True)
    filter_col1, filter_col2, filter_col3 = st.columns([1.2, 2, 1.2])
    with filter_col1:
        selected_category = st.selectbox("카테고리", ["전체"] + category_summary["카테고리"].tolist(), key="dashboard_category")
    with filter_col2:
        keyword = st.text_input("검색어", placeholder="아이폰, 책상, 자전거...", key="dashboard_keyword")
    with filter_col3:
        sort_label = st.selectbox("정렬", ["낮은 가격순", "높은 가격순", "최근 수집순"], key="dashboard_sort")

    filtered = df.copy()
    if selected_category != "전체":
        filtered = filtered[filtered["카테고리"] == selected_category]
    if keyword:
        filtered = filtered[filtered["제목"].astype(str).str.contains(keyword, case=False, na=False)]

    if sort_label == "낮은 가격순":
        sorted_items = filtered.sort_values("가격_numeric", ascending=True)
    elif sort_label == "높은 가격순":
        sorted_items = filtered.sort_values("가격_numeric", ascending=False)
    else:
        sorted_items = filtered.sort_index(ascending=False)

    st.caption(f"{len(filtered):,}개 상품")
    for _, row in sorted_items.head(12).iterrows():
        title = escape(str(row.get("제목", "")))
        category = escape(str(row.get("카테고리", "")))
        town = escape(str(row.get("수집동네", "")))
        price = escape(str(row.get("가격", format_won(row.get("가격_numeric")))))
        link = escape(str(row.get("게시글링크", "")))
        link_html = f' · <a href="{link}" target="_blank">게시글 보기</a>' if link.startswith("http") else ""
        st.markdown(
            f"""
            <div class="item-row">
                <div class="item-title">{title}</div>
                <div class="item-meta">
                    <span class="item-price">{price}</span> · {category} · {town}{link_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("전체 데이터 표로 보기"):
        st.dataframe(
            filtered[["수집동네", "제목", "가격", "카테고리", "게시글링크"]].head(200),
            use_container_width=True,
        )


def main():
    load_dotenv_if_exists()
    inject_theme_style()

    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "링크 분석"

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">D</div>
            <div class="sidebar-title">유학생<br>거래 도우미</div>
            <div class="sidebar-copy">가격 확인부터 외국어 메시지까지 한 번에 도와드려요.</div>
        </div>
        <div class="sidebar-label">QUICK</div>
        """,
        unsafe_allow_html=True,
    )

    quick_col1, quick_col2 = st.sidebar.columns(2)
    with quick_col1:
        if st.button("🏠 처음", use_container_width=True):
            reset_inputs()
            st.session_state["nav_page"] = "링크 분석"
            st.rerun()
    with quick_col2:
        if st.button("↻ 새로", use_container_width=True):
            reset_inputs()
            st.rerun()

    st.sidebar.markdown('<div class="sidebar-label">MENU</div>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        "메뉴",
        ["링크 분석", "직접 입력", "데이터 대시보드"],
        label_visibility="collapsed",
        key="nav_page",
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

    if page == "링크 분석":
        render_analysis_page()
    elif page == "직접 입력":
        render_manual_page()
    else:
        render_dashboard_page()


if __name__ == "__main__":
    main()
