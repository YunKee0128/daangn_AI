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


LANG_CODE_MAP = {
    "한국어": "ko",
    "English": "en",
    "日本語": "ja",
    "中文": "zh",
    "Tiếng Việt": "vi",
    "ไทย": "th",
}
CODE_TO_LANG_LABEL = {code: label for label, code in LANG_CODE_MAP.items()}

STATUS_KEY_MAP = {
    "저렴": "cheap",
    "적정": "fair",
    "비쌈": "expensive",
    "가격 입력 필요": "need_price",
    "판단 불가": "unknown",
}

NAV_KEY_TO_TKEY = {
    "링크 분석": "nav_link_analysis",
    "직접 입력": "nav_manual_input",
    "데이터 대시보드": "nav_dashboard",
    "메시지 번역": "nav_translate",
}

CATEGORY_LABELS_I18N = {
    "ko": {"전자기기": "전자기기", "가구": "가구", "생활가전": "생활가전", "스포츠": "스포츠", "도서": "도서", "패션": "패션", "기타": "기타"},
    "en": {"전자기기": "Electronics", "가구": "Furniture", "생활가전": "Home Appliances", "스포츠": "Sports", "도서": "Books", "패션": "Fashion", "기타": "Other"},
    "ja": {"전자기기": "電子機器", "가구": "家具", "생활가전": "生活家電", "스포츠": "スポーツ", "도서": "本", "패션": "ファッション", "기타": "その他"},
    "zh": {"전자기기": "电子产品", "가구": "家具", "생활가전": "生活家电", "스포츠": "运动用品", "도서": "图书", "패션": "时尚", "기타": "其他"},
    "vi": {"전자기기": "Đồ điện tử", "가구": "Nội thất", "생활가전": "Đồ gia dụng", "스포츠": "Thể thao", "도서": "Sách", "패션": "Thời trang", "기타": "Khác"},
    "th": {"전자기기": "อุปกรณ์อิเล็กทรอนิกส์", "가구": "เฟอร์นิเจอร์", "생활가전": "เครื่องใช้ไฟฟ้า", "스포츠": "กีฬา", "도서": "หนังสือ", "패션": "แฟชั่น", "기타": "อื่นๆ"},
}

TRANSLATIONS = {
    "ko": {
        "ui_language_label": "언어 (UI)",
        "brand_title_html": "유학생<br>거래 도우미",
        "brand_copy": "가격 확인부터 외국어 메시지까지 한 번에 도와드려요.",
        "refresh_button": "↻ 새로고침",
        "menu_label": "MENU",
        "home_help": "첫 화면으로 돌아가기",
        "status_model_ready": "시세 모델 준비 완료",
        "status_model_missing": "시세 모델을 찾을 수 없어요",
        "status_ai_key_missing": "구매 도움말은 키 설정 후 사용 가능",
        "nav_link_analysis": "링크 분석",
        "nav_manual_input": "직접 입력",
        "nav_dashboard": "데이터 대시보드",
        "nav_translate": "메시지 번역",
        "app_kicker": "부산외국어대학교 유학생 중고거래 도우미",
        "price_check_result": "가격 체크 결과",
        "actual_price": "판매가",
        "expected_price": "예상 시세",
        "error_info_prefix": "오류 정보: ",
        "status_badge_cheap": "저렴", "status_title_cheap": "좋은 거래일 가능성이 높아요",
        "status_copy_cheap": "판매가가 예상 시세보다 낮은 편입니다. 다만 너무 저렴한 상품은 상태, 구성품, 직거래 장소를 한 번 더 확인하세요.",
        "status_badge_fair": "적정", "status_title_fair": "무난한 가격대로 보여요",
        "status_copy_fair": "판매가가 예상 시세와 크게 차이 나지 않습니다. 상품 상태와 구성품이 설명과 맞는지 확인하면 좋아요.",
        "status_badge_expensive": "비쌈", "status_title_expensive": "가격이 높은 편이에요",
        "status_copy_expensive": "판매가가 예상 시세보다 높게 나왔습니다. 같은 상품의 다른 매물과 비교하거나 가격 조정을 요청해보세요.",
        "status_badge_need_price": "가격 입력 필요", "status_title_need_price": "판매가를 입력하면 판단할 수 있어요",
        "status_copy_need_price": "게시글에서 가격을 읽지 못했습니다. 판매가를 직접 입력하면 예상 시세와 비교해드릴게요.",
        "status_badge_unknown": "판단 불가", "status_title_unknown": "지금은 판단하기 어려워요",
        "status_copy_unknown": "모델이나 가격 정보가 부족합니다. 상품명과 가격을 더 구체적으로 입력해보세요.",
        "link_hero_title": "링크만 붙여넣으면 끝",
        "link_hero_subtitle": "판매가가 괜찮은지 바로 확인하고, 외국어로 보낼 메시지까지 한 번에 준비하세요.",
        "link_step1_title": "링크 붙여넣기", "link_step1_copy": "당근 게시글 주소를 그대로 넣어요.",
        "link_step2_title": "판매가 확인", "link_step2_copy": "자동 인식이 안 되면 직접 입력해요.",
        "link_step3_title": "구매 판단", "link_step3_copy": "예상 시세와 거래 메시지를 확인해요.",
        "link_form1_heading": "상품 링크", "link_form1_note": "당근마켓 게시글 주소를 복사해서 붙여넣으세요.",
        "link_form2_heading": "상품 종류", "link_form2_note": "링크만으로 종류를 알기 어려울 때를 위해 먼저 골라주세요.",
        "option_row_title": "확인 옵션",
        "link_checkbox_manual_price": "판매가 직접 입력",
        "price_label": "판매가",
        "language_select_label": "언어",
        "link_checkbox_use_ai": "도움말 받기",
        "link_submit": "상품 확인하기",
        "link_spinner_analyzing": "게시글을 읽고 가격을 분석하는 중입니다...",
        "manual_spinner_generating": "구매 도움말을 만드는 중입니다...",
        "notice_need_link_title": "상품 링크를 넣어주세요",
        "notice_need_link_copy": "당근마켓 게시글 주소를 복사해서 붙여넣으면 바로 확인할 수 있어요.",
        "notice_link_error_title": "링크를 읽지 못했어요",
        "notice_link_error_copy": "게시글 접근이 제한되었거나 주소가 올바르지 않을 수 있어요. 같은 상품을 직접 입력 메뉴에서 확인해보세요.",
        "section_item_info": "상품 정보",
        "info_title": "상품명", "info_price": "판매가", "info_category": "분류", "info_expected_price": "예상 시세",
        "expander_seller_desc": "판매자 설명",
        "section_purchase_help": "구매 도움말",
        "manual_hero_title": "무슨 물건인지 먼저 골라주세요",
        "manual_hero_subtitle": "상품 종류를 고르면 입력 예시가 바뀌고, 더 자연스럽게 가격을 확인할 수 있어요.",
        "manual_form1_heading": "상품 종류", "manual_form1_note": "가장 가까운 종류를 먼저 선택하세요. 정확히 몰라도 괜찮아요.",
        "manual_form2_heading": "상품 정보", "manual_form2_note": "제목과 판매가만 넣어도 확인할 수 있어요. 설명을 넣으면 구매 도움말이 더 좋아집니다.",
        "manual_title_label": "상품명", "manual_price_label": "판매가", "manual_help_language_label": "도움말 언어",
        "manual_desc_label": "판매자 설명",
        "manual_desc_placeholder": "예: 사용 기간, 구성품, 하자 여부, 직거래 위치 등을 적어주세요.",
        "manual_option_row_title": "추가 옵션",
        "manual_checkbox_use_ai": "구매 도움말 받기",
        "manual_submit": "가격 확인하기",
        "notice_need_title_title": "상품명을 입력해주세요",
        "notice_need_title_copy": "예: 아이폰 13, 책상, 자전거처럼 상품을 알아볼 수 있게 적으면 더 정확해져요.",
        "section_price_analysis": "가격 분석 결과",
        "price_disclaimer": "예상 시세는 앱이 가진 거래 데이터로 계산한 값입니다. 실제 거래 전에는 상품 상태와 구성품을 함께 확인하세요.",
        "dashboard_hero_title": "부산 중고거래 시세 보기",
        "dashboard_hero_subtitle": "수집한 상품 데이터를 카테고리별로 훑어보고, 궁금한 물건을 바로 검색해보세요.",
        "dashboard_no_data": "used_items_cleaned.csv 파일을 찾을 수 없습니다.",
        "section_overview": "한눈에 보기",
        "overview_total_items": "등록 상품", "overview_avg_price": "평균 시세", "overview_median_price": "중간 가격", "overview_category_count": "상품 종류",
        "section_category_price": "카테고리별 시세",
        "category_stat_template": "{count:,}개 상품 · 평균 {avg}",
        "section_browse_items": "상품 찾아보기",
        "filter_category_label": "카테고리", "filter_keyword_label": "검색어", "filter_sort_label": "정렬",
        "sort_price_asc": "낮은 가격순", "sort_price_desc": "높은 가격순", "sort_recent": "최근 수집순",
        "all_option": "전체",
        "item_count_template": "{count:,}개 상품",
        "view_post_link": "게시글 보기",
        "expander_full_table": "전체 데이터 표로 보기",
        "col_town": "수집동네", "col_title": "제목", "col_price": "가격", "col_category": "카테고리", "col_link": "게시글링크",
        "translate_hero_title": "판매자에게 보낼 말, 한국어로 바꿔드려요",
        "translate_hero_subtitle": "가격 조회 없이도 언제든 메시지만 번역해서 바로 복사해 쓸 수 있어요.",
        "translate_source_lang_label": "내 언어",
        "translate_input_label": "내가 쓴 메시지",
        "translate_input_placeholder": "예: Is this item still available? Can we meet near the school tomorrow?",
        "translate_button": "한국어로 번역하기",
        "translating_spinner": "번역하는 중입니다...",
        "translate_empty_message": "번역할 메시지를 입력해주세요.",
        "openai_key_missing_help": "OpenAI API Key가 없어 구매 도움말을 만들지 못했습니다.",
        "openai_pkg_missing_help": "openai 패키지가 설치되어 있지 않아 구매 도움말을 만들지 못했습니다.",
        "openai_key_missing_translate": "OpenAI API Key가 없어 번역하지 못했습니다.",
        "openai_pkg_missing_translate": "openai 패키지가 설치되어 있지 않아 번역하지 못했습니다.",
    },
    "en": {
        "ui_language_label": "UI Language",
        "brand_title_html": "Student<br>Trade Helper",
        "brand_copy": "From price checks to foreign-language messages, all in one place.",
        "refresh_button": "↻ Refresh",
        "menu_label": "MENU",
        "home_help": "Back to home",
        "status_model_ready": "Price model ready",
        "status_model_missing": "Price model not found",
        "status_ai_key_missing": "Purchase help needs an API key",
        "nav_link_analysis": "Link Analysis",
        "nav_manual_input": "Manual Entry",
        "nav_dashboard": "Data Dashboard",
        "nav_translate": "Translate Message",
        "app_kicker": "Busan University of Foreign Studies — Used Goods Helper",
        "price_check_result": "Price Check Result",
        "actual_price": "Listed Price",
        "expected_price": "Expected Price",
        "error_info_prefix": "Error details: ",
        "status_badge_cheap": "Cheap", "status_title_cheap": "Looks like a good deal",
        "status_copy_cheap": "The listed price is lower than expected. If it seems unusually cheap, double-check the condition, included items, and meeting location.",
        "status_badge_fair": "Fair", "status_title_fair": "Looks like a reasonable price",
        "status_copy_fair": "The listed price is close to the expected price. Still worth confirming the item's condition and included items.",
        "status_badge_expensive": "Pricey", "status_title_expensive": "The price is on the higher side",
        "status_copy_expensive": "The listed price is higher than expected. Compare with other listings of the same item, or ask for a lower price.",
        "status_badge_need_price": "Price Needed", "status_title_need_price": "Enter the price to get a judgment",
        "status_copy_need_price": "We couldn't read a price from the post. Enter it yourself and we'll compare it to the expected price.",
        "status_badge_unknown": "Unable to Judge", "status_title_unknown": "Hard to judge right now",
        "status_copy_unknown": "There isn't enough model or price information. Try entering a more specific title and price.",
        "link_hero_title": "Just paste the link",
        "link_hero_subtitle": "Check if the price is fair and get a ready-to-send message in another language, all at once.",
        "link_step1_title": "Paste the link", "link_step1_copy": "Paste the Karrot (당근마켓) post URL as-is.",
        "link_step2_title": "Check the price", "link_step2_copy": "If it's not detected automatically, enter it yourself.",
        "link_step3_title": "Decide whether to buy", "link_step3_copy": "Check the expected price and the chat message.",
        "link_form1_heading": "Item Link", "link_form1_note": "Copy and paste the Karrot post URL.",
        "link_form2_heading": "Item Type", "link_form2_note": "Pick this first in case the type can't be guessed from the link alone.",
        "option_row_title": "Options",
        "link_checkbox_manual_price": "Enter price manually",
        "price_label": "Price",
        "language_select_label": "Language",
        "link_checkbox_use_ai": "Get purchase help",
        "link_submit": "Check This Item",
        "link_spinner_analyzing": "Reading the post and analyzing the price...",
        "manual_spinner_generating": "Generating purchase help...",
        "notice_need_link_title": "Please enter an item link",
        "notice_need_link_copy": "Paste the Karrot post URL to check it right away.",
        "notice_link_error_title": "Couldn't read the link",
        "notice_link_error_copy": "The post might be restricted or the URL might be wrong. Try checking the same item in Manual Entry instead.",
        "section_item_info": "Item Info",
        "info_title": "Item Name", "info_price": "Listed Price", "info_category": "Category", "info_expected_price": "Expected Price",
        "expander_seller_desc": "Seller's Description",
        "section_purchase_help": "Purchase Help",
        "manual_hero_title": "First, pick what kind of item it is",
        "manual_hero_subtitle": "Choosing a category changes the examples and gives a more natural price check.",
        "manual_form1_heading": "Item Type", "manual_form1_note": "Pick whichever is closest — it doesn't need to be exact.",
        "manual_form2_heading": "Item Info", "manual_form2_note": "A title and price are enough. Adding a description makes the purchase help better.",
        "manual_title_label": "Item Name", "manual_price_label": "Price", "manual_help_language_label": "Help Language",
        "manual_desc_label": "Seller's Description",
        "manual_desc_placeholder": "e.g. how long it was used, what's included, any defects, meeting location, etc.",
        "manual_option_row_title": "Additional Options",
        "manual_checkbox_use_ai": "Get purchase help",
        "manual_submit": "Check Price",
        "notice_need_title_title": "Please enter an item name",
        "notice_need_title_copy": "e.g. iPhone 13, desk, bicycle — a clear name gives a more accurate result.",
        "section_price_analysis": "Price Analysis Result",
        "price_disclaimer": "The expected price is calculated from the app's collected trade data. Always check the item's condition and included items before the actual trade.",
        "dashboard_hero_title": "Browse Busan resale prices",
        "dashboard_hero_subtitle": "Browse collected item data by category, or search for something specific.",
        "dashboard_no_data": "Couldn't find used_items_cleaned.csv.",
        "section_overview": "Overview",
        "overview_total_items": "Listed Items", "overview_avg_price": "Average Price", "overview_median_price": "Median Price", "overview_category_count": "Item Types",
        "section_category_price": "Prices by Category",
        "category_stat_template": "{count:,} items · avg {avg}",
        "section_browse_items": "Browse Items",
        "filter_category_label": "Category", "filter_keyword_label": "Search", "filter_sort_label": "Sort",
        "sort_price_asc": "Price: Low to High", "sort_price_desc": "Price: High to Low", "sort_recent": "Recently Collected",
        "all_option": "All",
        "item_count_template": "{count:,} items",
        "view_post_link": "View Post",
        "expander_full_table": "View Full Data Table",
        "col_town": "Location", "col_title": "Title", "col_price": "Price", "col_category": "Category", "col_link": "Post Link",
        "translate_hero_title": "We'll turn your message into Korean",
        "translate_hero_subtitle": "Translate a message anytime, even without a price check, and copy it right away.",
        "translate_source_lang_label": "My Language",
        "translate_input_label": "My Message",
        "translate_input_placeholder": "e.g. Is this item still available? Can we meet near the school tomorrow?",
        "translate_button": "Translate to Korean",
        "translating_spinner": "Translating...",
        "translate_empty_message": "Please enter a message to translate.",
        "openai_key_missing_help": "No OpenAI API key is set, so purchase help couldn't be generated.",
        "openai_pkg_missing_help": "The openai package isn't installed, so purchase help couldn't be generated.",
        "openai_key_missing_translate": "No OpenAI API key is set, so the message couldn't be translated.",
        "openai_pkg_missing_translate": "The openai package isn't installed, so the message couldn't be translated.",
    },
    "ja": {
        "ui_language_label": "表示言語",
        "brand_title_html": "留学生<br>取引アシスタント",
        "brand_copy": "価格チェックから外国語メッセージまで、まとめてサポートします。",
        "refresh_button": "↻ 更新",
        "menu_label": "メニュー",
        "home_help": "最初の画面に戻る",
        "status_model_ready": "価格モデル準備完了",
        "status_model_missing": "価格モデルが見つかりません",
        "status_ai_key_missing": "購入ヘルプはキー設定後に利用可能",
        "nav_link_analysis": "リンク分析",
        "nav_manual_input": "直接入力",
        "nav_dashboard": "データダッシュボード",
        "nav_translate": "メッセージ翻訳",
        "app_kicker": "釜山外国語大学 留学生中古取引アシスタント",
        "price_check_result": "価格チェック結果",
        "actual_price": "販売価格",
        "expected_price": "予想相場",
        "error_info_prefix": "エラー情報: ",
        "status_badge_cheap": "お得", "status_title_cheap": "良い取引の可能性が高いです",
        "status_copy_cheap": "販売価格が予想相場より低めです。ただし安すぎる場合は状態・付属品・取引場所を必ず確認しましょう。",
        "status_badge_fair": "適正", "status_title_fair": "妥当な価格帯に見えます",
        "status_copy_fair": "販売価格は予想相場と大きく変わりません。商品の状態や付属品が説明と合っているか確認しましょう。",
        "status_badge_expensive": "高め", "status_title_expensive": "価格が高めです",
        "status_copy_expensive": "販売価格が予想相場より高くなっています。同じ商品の他の出品と比べるか、値下げを相談してみましょう。",
        "status_badge_need_price": "価格入力が必要", "status_title_need_price": "価格を入力すると判定できます",
        "status_copy_need_price": "投稿から価格を読み取れませんでした。価格を直接入力すると予想相場と比較します。",
        "status_badge_unknown": "判定不可", "status_title_unknown": "現時点では判定が難しいです",
        "status_copy_unknown": "モデルまたは価格情報が不足しています。商品名と価格をより具体的に入力してみてください。",
        "link_hero_title": "リンクを貼るだけ",
        "link_hero_subtitle": "価格が妥当か確認して、外国語で送るメッセージも一度に準備しましょう。",
        "link_step1_title": "リンクを貼る", "link_step1_copy": "당근(カロット)の投稿URLをそのまま入力します。",
        "link_step2_title": "価格を確認", "link_step2_copy": "自動で読み取れない場合は直接入力します。",
        "link_step3_title": "購入を判断", "link_step3_copy": "予想相場と取引メッセージを確認します。",
        "link_form1_heading": "商品リンク", "link_form1_note": "당근마켓の投稿URLをコピーして貼り付けてください。",
        "link_form2_heading": "商品の種類", "link_form2_note": "リンクだけでは種類が分かりにくい場合のために先に選んでください。",
        "option_row_title": "確認オプション",
        "link_checkbox_manual_price": "販売価格を直接入力",
        "price_label": "販売価格",
        "language_select_label": "言語",
        "link_checkbox_use_ai": "ヘルプを受け取る",
        "link_submit": "商品を確認する",
        "link_spinner_analyzing": "投稿を読み込んで価格を分析しています...",
        "manual_spinner_generating": "購入ヘルプを作成しています...",
        "notice_need_link_title": "商品リンクを入力してください",
        "notice_need_link_copy": "당근마켓の投稿URLをコピーして貼り付けるとすぐに確認できます。",
        "notice_link_error_title": "リンクを読み取れませんでした",
        "notice_link_error_copy": "投稿へのアクセスが制限されているか、URLが正しくない可能性があります。同じ商品を直接入力メニューで確認してみてください。",
        "section_item_info": "商品情報",
        "info_title": "商品名", "info_price": "販売価格", "info_category": "分類", "info_expected_price": "予想相場",
        "expander_seller_desc": "出品者の説明",
        "section_purchase_help": "購入ヘルプ",
        "manual_hero_title": "まずは商品の種類を選んでください",
        "manual_hero_subtitle": "商品の種類を選ぶと入力例が変わり、より自然に価格を確認できます。",
        "manual_form1_heading": "商品の種類", "manual_form1_note": "最も近い種類を選んでください。正確でなくても大丈夫です。",
        "manual_form2_heading": "商品情報", "manual_form2_note": "商品名と価格だけでも確認できます。説明を入れると購入ヘルプの質が上がります。",
        "manual_title_label": "商品名", "manual_price_label": "販売価格", "manual_help_language_label": "ヘルプの言語",
        "manual_desc_label": "出品者の説明",
        "manual_desc_placeholder": "例: 使用期間、付属品、不具合の有無、取引場所などを書いてください。",
        "manual_option_row_title": "追加オプション",
        "manual_checkbox_use_ai": "購入ヘルプを受け取る",
        "manual_submit": "価格を確認する",
        "notice_need_title_title": "商品名を入力してください",
        "notice_need_title_copy": "例: iPhone 13、机、自転車のように分かりやすく書くとより正確になります。",
        "section_price_analysis": "価格分析結果",
        "price_disclaimer": "予想相場はアプリが持つ取引データから計算した値です。実際の取引前には商品の状態と付属品も確認してください。",
        "dashboard_hero_title": "釜山の中古取引相場を見る",
        "dashboard_hero_subtitle": "収集した商品データをカテゴリー別に確認したり、気になる商品を検索できます。",
        "dashboard_no_data": "used_items_cleaned.csv が見つかりません。",
        "section_overview": "概要",
        "overview_total_items": "登録商品数", "overview_avg_price": "平均相場", "overview_median_price": "中央値", "overview_category_count": "商品の種類数",
        "section_category_price": "カテゴリー別相場",
        "category_stat_template": "{count:,}件 · 平均 {avg}",
        "section_browse_items": "商品を探す",
        "filter_category_label": "カテゴリー", "filter_keyword_label": "検索キーワード", "filter_sort_label": "並び替え",
        "sort_price_asc": "価格が低い順", "sort_price_desc": "価格が高い順", "sort_recent": "最近収集した順",
        "all_option": "すべて",
        "item_count_template": "{count:,}件",
        "view_post_link": "投稿を見る",
        "expander_full_table": "全データを表で見る",
        "col_town": "収集地域", "col_title": "タイトル", "col_price": "価格", "col_category": "カテゴリー", "col_link": "投稿リンク",
        "translate_hero_title": "送りたいメッセージを韓国語にします",
        "translate_hero_subtitle": "価格チェックなしでも、いつでもメッセージを翻訳してすぐコピーできます。",
        "translate_source_lang_label": "自分の言語",
        "translate_input_label": "書いたメッセージ",
        "translate_input_placeholder": "例: この商品はまだありますか？明日学校の近くで会えますか？",
        "translate_button": "韓国語に翻訳する",
        "translating_spinner": "翻訳しています...",
        "translate_empty_message": "翻訳するメッセージを入力してください。",
        "openai_key_missing_help": "OpenAI APIキーが設定されていないため購入ヘルプを作成できませんでした。",
        "openai_pkg_missing_help": "openaiパッケージがインストールされていないため購入ヘルプを作成できませんでした。",
        "openai_key_missing_translate": "OpenAI APIキーが設定されていないため翻訳できませんでした。",
        "openai_pkg_missing_translate": "openaiパッケージがインストールされていないため翻訳できませんでした。",
    },
    "zh": {
        "ui_language_label": "界面语言",
        "brand_title_html": "留学生<br>交易助手",
        "brand_copy": "从价格核实到外语消息，一站式帮您搞定。",
        "refresh_button": "↻ 刷新",
        "menu_label": "菜单",
        "home_help": "返回首页",
        "status_model_ready": "价格模型已就绪",
        "status_model_missing": "找不到价格模型",
        "status_ai_key_missing": "设置密钥后才能使用购买帮助",
        "nav_link_analysis": "链接分析",
        "nav_manual_input": "手动输入",
        "nav_dashboard": "数据看板",
        "nav_translate": "消息翻译",
        "app_kicker": "釜山外国语大学 留学生二手交易助手",
        "price_check_result": "价格核实结果",
        "actual_price": "售价",
        "expected_price": "预估行情",
        "error_info_prefix": "错误信息：",
        "status_badge_cheap": "便宜", "status_title_cheap": "很可能是划算的交易",
        "status_copy_cheap": "售价低于预估行情。不过如果价格过低，请再次确认商品状态、配件和交易地点。",
        "status_badge_fair": "合理", "status_title_fair": "价格看起来比较合理",
        "status_copy_fair": "售价与预估行情相差不大。建议确认商品状态和配件是否与描述一致。",
        "status_badge_expensive": "偏贵", "status_title_expensive": "价格偏高",
        "status_copy_expensive": "售价高于预估行情。可以对比同款商品的其他卖家，或尝试请卖家调整价格。",
        "status_badge_need_price": "需要输入价格", "status_title_need_price": "输入售价即可判断",
        "status_copy_need_price": "未能从帖子中读取到价格。请直接输入售价，我们会与预估行情进行比较。",
        "status_badge_unknown": "暂无法判断", "status_title_unknown": "目前难以判断",
        "status_copy_unknown": "模型或价格信息不足。请尝试输入更具体的商品名称和价格。",
        "link_hero_title": "只需粘贴链接",
        "link_hero_subtitle": "立即确认价格是否合理，并一次性准备好外语消息。",
        "link_step1_title": "粘贴链接", "link_step1_copy": "直接粘贴当根（Karrot）帖子链接。",
        "link_step2_title": "确认售价", "link_step2_copy": "若无法自动识别，请手动输入。",
        "link_step3_title": "判断是否购买", "link_step3_copy": "查看预估行情和交易消息。",
        "link_form1_heading": "商品链接", "link_form1_note": "复制并粘贴当根市场帖子的链接。",
        "link_form2_heading": "商品类型", "link_form2_note": "若仅凭链接难以判断类型，请先在此选择。",
        "option_row_title": "确认选项",
        "link_checkbox_manual_price": "手动输入售价",
        "price_label": "售价",
        "language_select_label": "语言",
        "link_checkbox_use_ai": "获取帮助说明",
        "link_submit": "确认商品",
        "link_spinner_analyzing": "正在读取帖子并分析价格...",
        "manual_spinner_generating": "正在生成购买帮助...",
        "notice_need_link_title": "请输入商品链接",
        "notice_need_link_copy": "复制并粘贴当根市场帖子链接即可立即查看。",
        "notice_link_error_title": "无法读取该链接",
        "notice_link_error_copy": "帖子访问可能受限，或链接地址有误。可以尝试在“手动输入”菜单中查看同一商品。",
        "section_item_info": "商品信息",
        "info_title": "商品名称", "info_price": "售价", "info_category": "分类", "info_expected_price": "预估行情",
        "expander_seller_desc": "卖家描述",
        "section_purchase_help": "购买帮助",
        "manual_hero_title": "请先选择商品类型",
        "manual_hero_subtitle": "选择商品类型后，输入示例会随之变化，价格核实也会更准确。",
        "manual_form1_heading": "商品类型", "manual_form1_note": "请先选择最接近的类型，不必完全准确。",
        "manual_form2_heading": "商品信息", "manual_form2_note": "只填写标题和售价也能查看结果。补充描述会让购买帮助更完善。",
        "manual_title_label": "商品名称", "manual_price_label": "售价", "manual_help_language_label": "帮助说明语言",
        "manual_desc_label": "卖家描述",
        "manual_desc_placeholder": "例如：使用时长、附带物品、是否有瑕疵、交易地点等。",
        "manual_option_row_title": "更多选项",
        "manual_checkbox_use_ai": "获取购买帮助",
        "manual_submit": "确认价格",
        "notice_need_title_title": "请输入商品名称",
        "notice_need_title_copy": "例如 iPhone 13、书桌、自行车，写得越具体结果越准确。",
        "section_price_analysis": "价格分析结果",
        "price_disclaimer": "预估行情是根据应用收集的交易数据计算得出的。实际交易前请务必确认商品状态和配件。",
        "dashboard_hero_title": "查看釜山二手交易行情",
        "dashboard_hero_subtitle": "按类别浏览已收集的商品数据，也可以直接搜索感兴趣的商品。",
        "dashboard_no_data": "找不到 used_items_cleaned.csv 文件。",
        "section_overview": "总览",
        "overview_total_items": "已收录商品", "overview_avg_price": "平均行情", "overview_median_price": "中位价格", "overview_category_count": "商品类型数",
        "section_category_price": "各类别行情",
        "category_stat_template": "{count:,}件商品 · 平均 {avg}",
        "section_browse_items": "浏览商品",
        "filter_category_label": "类别", "filter_keyword_label": "搜索关键词", "filter_sort_label": "排序",
        "sort_price_asc": "价格从低到高", "sort_price_desc": "价格从高到低", "sort_recent": "最近收集顺序",
        "all_option": "全部",
        "item_count_template": "{count:,}件商品",
        "view_post_link": "查看帖子",
        "expander_full_table": "查看完整数据表",
        "col_town": "收集地区", "col_title": "标题", "col_price": "价格", "col_category": "类别", "col_link": "帖子链接",
        "translate_hero_title": "帮你把要发的话翻译成韩语",
        "translate_hero_subtitle": "无需先核实价格，随时都能翻译消息并立即复制使用。",
        "translate_source_lang_label": "我的语言",
        "translate_input_label": "我写的消息",
        "translate_input_placeholder": "例如：这个商品还在吗？明天可以在学校附近见面吗？",
        "translate_button": "翻译成韩语",
        "translating_spinner": "翻译中...",
        "translate_empty_message": "请输入需要翻译的消息。",
        "openai_key_missing_help": "未设置 OpenAI API 密钥，无法生成购买帮助。",
        "openai_pkg_missing_help": "未安装 openai 软件包，无法生成购买帮助。",
        "openai_key_missing_translate": "未设置 OpenAI API 密钥，无法进行翻译。",
        "openai_pkg_missing_translate": "未安装 openai 软件包，无法进行翻译。",
    },
    "vi": {
        "ui_language_label": "Ngôn ngữ giao diện",
        "brand_title_html": "Trợ lý<br>giao dịch du học sinh",
        "brand_copy": "Từ kiểm tra giá đến soạn tin nhắn ngoại ngữ, hỗ trợ bạn trong một nơi.",
        "refresh_button": "↻ Làm mới",
        "menu_label": "MENU",
        "home_help": "Quay về màn hình đầu",
        "status_model_ready": "Mô hình giá đã sẵn sàng",
        "status_model_missing": "Không tìm thấy mô hình giá",
        "status_ai_key_missing": "Cần thiết lập API key để dùng gợi ý mua hàng",
        "nav_link_analysis": "Phân tích liên kết",
        "nav_manual_input": "Nhập thủ công",
        "nav_dashboard": "Bảng dữ liệu",
        "nav_translate": "Dịch tin nhắn",
        "app_kicker": "Trợ lý mua bán đồ cũ cho du học sinh — Đại học Ngoại ngữ Busan",
        "price_check_result": "Kết quả kiểm tra giá",
        "actual_price": "Giá bán",
        "expected_price": "Giá dự đoán",
        "error_info_prefix": "Thông tin lỗi: ",
        "status_badge_cheap": "Rẻ", "status_title_cheap": "Có khả năng là món hời",
        "status_copy_cheap": "Giá bán thấp hơn giá dự đoán. Nếu quá rẻ, hãy kiểm tra kỹ tình trạng, phụ kiện đi kèm và địa điểm giao dịch.",
        "status_badge_fair": "Hợp lý", "status_title_fair": "Mức giá có vẻ hợp lý",
        "status_copy_fair": "Giá bán không chênh lệch nhiều so với giá dự đoán. Nên kiểm tra tình trạng và phụ kiện có đúng như mô tả không.",
        "status_badge_expensive": "Đắt", "status_title_expensive": "Giá hơi cao",
        "status_copy_expensive": "Giá bán cao hơn giá dự đoán. Hãy so sánh với các tin đăng khác của cùng sản phẩm hoặc thử thương lượng giá.",
        "status_badge_need_price": "Cần nhập giá", "status_title_need_price": "Nhập giá bán để được đánh giá",
        "status_copy_need_price": "Không đọc được giá từ bài đăng. Hãy nhập giá bán để so sánh với giá dự đoán.",
        "status_badge_unknown": "Chưa thể đánh giá", "status_title_unknown": "Hiện chưa thể đánh giá",
        "status_copy_unknown": "Thiếu dữ liệu mô hình hoặc giá. Hãy thử nhập tên sản phẩm và giá cụ thể hơn.",
        "link_hero_title": "Chỉ cần dán liên kết",
        "link_hero_subtitle": "Kiểm tra ngay giá có hợp lý không, và chuẩn bị luôn tin nhắn bằng ngoại ngữ.",
        "link_step1_title": "Dán liên kết", "link_step1_copy": "Dán nguyên địa chỉ bài đăng Karrot (당근마켓).",
        "link_step2_title": "Kiểm tra giá bán", "link_step2_copy": "Nếu không tự nhận diện được, hãy nhập tay.",
        "link_step3_title": "Quyết định mua", "link_step3_copy": "Xem giá dự đoán và tin nhắn giao dịch.",
        "link_form1_heading": "Liên kết sản phẩm", "link_form1_note": "Sao chép và dán địa chỉ bài đăng trên Karrot (당근마켓).",
        "link_form2_heading": "Loại sản phẩm", "link_form2_note": "Chọn trước phòng khi không thể đoán loại chỉ từ liên kết.",
        "option_row_title": "Tùy chọn kiểm tra",
        "link_checkbox_manual_price": "Nhập giá bán thủ công",
        "price_label": "Giá bán",
        "language_select_label": "Ngôn ngữ",
        "link_checkbox_use_ai": "Nhận gợi ý mua hàng",
        "link_submit": "Kiểm tra sản phẩm",
        "link_spinner_analyzing": "Đang đọc bài đăng và phân tích giá...",
        "manual_spinner_generating": "Đang tạo gợi ý mua hàng...",
        "notice_need_link_title": "Vui lòng nhập liên kết sản phẩm",
        "notice_need_link_copy": "Dán địa chỉ bài đăng Karrot để kiểm tra ngay.",
        "notice_link_error_title": "Không đọc được liên kết",
        "notice_link_error_copy": "Bài đăng có thể bị giới hạn truy cập hoặc địa chỉ không đúng. Hãy thử kiểm tra sản phẩm này ở mục Nhập thủ công.",
        "section_item_info": "Thông tin sản phẩm",
        "info_title": "Tên sản phẩm", "info_price": "Giá bán", "info_category": "Phân loại", "info_expected_price": "Giá dự đoán",
        "expander_seller_desc": "Mô tả của người bán",
        "section_purchase_help": "Gợi ý mua hàng",
        "manual_hero_title": "Trước tiên hãy chọn loại sản phẩm",
        "manual_hero_subtitle": "Chọn loại sản phẩm sẽ đổi ví dụ nhập liệu và giúp kiểm tra giá tự nhiên hơn.",
        "manual_form1_heading": "Loại sản phẩm", "manual_form1_note": "Chọn loại gần đúng nhất, không cần chính xác tuyệt đối.",
        "manual_form2_heading": "Thông tin sản phẩm", "manual_form2_note": "Chỉ cần tên và giá bán cũng có thể kiểm tra. Thêm mô tả sẽ giúp gợi ý mua hàng tốt hơn.",
        "manual_title_label": "Tên sản phẩm", "manual_price_label": "Giá bán", "manual_help_language_label": "Ngôn ngữ gợi ý",
        "manual_desc_label": "Mô tả của người bán",
        "manual_desc_placeholder": "VD: thời gian sử dụng, phụ kiện đi kèm, lỗi (nếu có), địa điểm giao dịch...",
        "manual_option_row_title": "Tùy chọn thêm",
        "manual_checkbox_use_ai": "Nhận gợi ý mua hàng",
        "manual_submit": "Kiểm tra giá",
        "notice_need_title_title": "Vui lòng nhập tên sản phẩm",
        "notice_need_title_copy": "VD: iPhone 13, bàn học, xe đạp — ghi rõ ràng sẽ cho kết quả chính xác hơn.",
        "section_price_analysis": "Kết quả phân tích giá",
        "price_disclaimer": "Giá dự đoán được tính từ dữ liệu giao dịch mà ứng dụng thu thập. Hãy luôn kiểm tra tình trạng và phụ kiện trước khi giao dịch thực tế.",
        "dashboard_hero_title": "Xem giá đồ cũ tại Busan",
        "dashboard_hero_subtitle": "Xem dữ liệu sản phẩm đã thu thập theo từng danh mục, hoặc tìm ngay món đồ bạn quan tâm.",
        "dashboard_no_data": "Không tìm thấy tệp used_items_cleaned.csv.",
        "section_overview": "Tổng quan",
        "overview_total_items": "Sản phẩm đã đăng", "overview_avg_price": "Giá trung bình", "overview_median_price": "Giá trung vị", "overview_category_count": "Số loại sản phẩm",
        "section_category_price": "Giá theo danh mục",
        "category_stat_template": "{count:,} sản phẩm · TB {avg}",
        "section_browse_items": "Tìm sản phẩm",
        "filter_category_label": "Danh mục", "filter_keyword_label": "Từ khóa", "filter_sort_label": "Sắp xếp",
        "sort_price_asc": "Giá thấp đến cao", "sort_price_desc": "Giá cao đến thấp", "sort_recent": "Thu thập gần đây",
        "all_option": "Tất cả",
        "item_count_template": "{count:,} sản phẩm",
        "view_post_link": "Xem bài đăng",
        "expander_full_table": "Xem toàn bộ bảng dữ liệu",
        "col_town": "Khu vực", "col_title": "Tiêu đề", "col_price": "Giá", "col_category": "Danh mục", "col_link": "Liên kết bài đăng",
        "translate_hero_title": "Dịch tin nhắn của bạn sang tiếng Hàn",
        "translate_hero_subtitle": "Không cần kiểm tra giá, bạn vẫn có thể dịch tin nhắn bất cứ lúc nào và sao chép ngay.",
        "translate_source_lang_label": "Ngôn ngữ của tôi",
        "translate_input_label": "Tin nhắn của tôi",
        "translate_input_placeholder": "VD: Sản phẩm này còn không ạ? Ngày mai mình gặp gần trường được không?",
        "translate_button": "Dịch sang tiếng Hàn",
        "translating_spinner": "Đang dịch...",
        "translate_empty_message": "Vui lòng nhập tin nhắn cần dịch.",
        "openai_key_missing_help": "Chưa thiết lập OpenAI API key nên không thể tạo gợi ý mua hàng.",
        "openai_pkg_missing_help": "Chưa cài đặt gói openai nên không thể tạo gợi ý mua hàng.",
        "openai_key_missing_translate": "Chưa thiết lập OpenAI API key nên không thể dịch.",
        "openai_pkg_missing_translate": "Chưa cài đặt gói openai nên không thể dịch.",
    },
    "th": {
        "ui_language_label": "ภาษาที่แสดง",
        "brand_title_html": "ผู้ช่วย<br>ซื้อขายสำหรับนักศึกษาต่างชาติ",
        "brand_copy": "ตั้งแต่เช็กราคาไปจนถึงข้อความภาษาต่างประเทศ ช่วยคุณได้ในที่เดียว",
        "refresh_button": "↻ รีเฟรช",
        "menu_label": "เมนู",
        "home_help": "กลับไปหน้าแรก",
        "status_model_ready": "โมเดลราคาพร้อมใช้งาน",
        "status_model_missing": "ไม่พบโมเดลราคา",
        "status_ai_key_missing": "ต้องตั้งค่าคีย์ก่อนใช้คำแนะนำการซื้อ",
        "nav_link_analysis": "วิเคราะห์ลิงก์",
        "nav_manual_input": "กรอกข้อมูลเอง",
        "nav_dashboard": "แดชบอร์ดข้อมูล",
        "nav_translate": "แปลข้อความ",
        "app_kicker": "ผู้ช่วยซื้อขายมือสองสำหรับนักศึกษาต่างชาติ — มหาวิทยาลัยภาษาต่างประเทศปูซาน",
        "price_check_result": "ผลการเช็กราคา",
        "actual_price": "ราคาขาย",
        "expected_price": "ราคาที่คาดการณ์",
        "error_info_prefix": "ข้อมูลข้อผิดพลาด: ",
        "status_badge_cheap": "ราคาถูก", "status_title_cheap": "มีแนวโน้มเป็นดีลที่ดี",
        "status_copy_cheap": "ราคาขายต่ำกว่าราคาที่คาดการณ์ไว้ แต่ถ้าถูกเกินไป ควรตรวจสอบสภาพสินค้า อุปกรณ์ที่แถม และสถานที่นัดรับอีกครั้ง",
        "status_badge_fair": "เหมาะสม", "status_title_fair": "ราคาดูเหมาะสมดี",
        "status_copy_fair": "ราคาขายไม่ต่างจากราคาที่คาดการณ์มากนัก ควรตรวจสอบสภาพสินค้าและอุปกรณ์ให้ตรงกับคำอธิบาย",
        "status_badge_expensive": "ราคาสูง", "status_title_expensive": "ราคาค่อนข้างสูง",
        "status_copy_expensive": "ราคาขายสูงกว่าราคาที่คาดการณ์ ลองเปรียบเทียบกับประกาศอื่นของสินค้าเดียวกัน หรือขอให้ผู้ขายลดราคา",
        "status_badge_need_price": "ต้องกรอกราคา", "status_title_need_price": "กรอกราคาขายเพื่อให้ประเมินได้",
        "status_copy_need_price": "ไม่สามารถอ่านราคาจากประกาศได้ กรอกราคาขายเองเพื่อเปรียบเทียบกับราคาที่คาดการณ์",
        "status_badge_unknown": "ยังประเมินไม่ได้", "status_title_unknown": "ตอนนี้ยังประเมินได้ยาก",
        "status_copy_unknown": "ข้อมูลโมเดลหรือราคายังไม่เพียงพอ ลองกรอกชื่อสินค้าและราคาให้ชัดเจนขึ้น",
        "link_hero_title": "แค่วางลิงก์ก็เสร็จ",
        "link_hero_subtitle": "เช็กได้ทันทีว่าราคาขายเหมาะสมไหม พร้อมเตรียมข้อความภาษาต่างประเทศไว้ในที่เดียว",
        "link_step1_title": "วางลิงก์", "link_step1_copy": "วางที่อยู่ประกาศ Karrot (당근마켓) ได้เลย",
        "link_step2_title": "เช็กราคาขาย", "link_step2_copy": "ถ้าระบบตรวจจับไม่ได้ ให้กรอกเอง",
        "link_step3_title": "ตัดสินใจซื้อ", "link_step3_copy": "ดูราคาที่คาดการณ์และข้อความสำหรับติดต่อ",
        "link_form1_heading": "ลิงก์สินค้า", "link_form1_note": "คัดลอกและวางที่อยู่ประกาศจาก Karrot (당근마켓)",
        "link_form2_heading": "ประเภทสินค้า", "link_form2_note": "เลือกไว้ก่อนเผื่อกรณีที่ดูประเภทจากลิงก์อย่างเดียวไม่ได้",
        "option_row_title": "ตัวเลือกการเช็ก",
        "link_checkbox_manual_price": "กรอกราคาขายเอง",
        "price_label": "ราคาขาย",
        "language_select_label": "ภาษา",
        "link_checkbox_use_ai": "รับคำแนะนำ",
        "link_submit": "เช็กสินค้านี้",
        "link_spinner_analyzing": "กำลังอ่านประกาศและวิเคราะห์ราคา...",
        "manual_spinner_generating": "กำลังสร้างคำแนะนำการซื้อ...",
        "notice_need_link_title": "กรุณากรอกลิงก์สินค้า",
        "notice_need_link_copy": "วางที่อยู่ประกาศ Karrot เพื่อเช็กได้ทันที",
        "notice_link_error_title": "ไม่สามารถอ่านลิงก์นี้ได้",
        "notice_link_error_copy": "ประกาศอาจถูกจำกัดการเข้าถึง หรือที่อยู่ไม่ถูกต้อง ลองเช็กสินค้าเดียวกันในเมนู กรอกข้อมูลเอง แทน",
        "section_item_info": "ข้อมูลสินค้า",
        "info_title": "ชื่อสินค้า", "info_price": "ราคาขาย", "info_category": "หมวดหมู่", "info_expected_price": "ราคาที่คาดการณ์",
        "expander_seller_desc": "คำอธิบายจากผู้ขาย",
        "section_purchase_help": "คำแนะนำการซื้อ",
        "manual_hero_title": "เลือกประเภทสินค้าก่อน",
        "manual_hero_subtitle": "การเลือกประเภทสินค้าจะเปลี่ยนตัวอย่างการกรอก และช่วยให้เช็กราคาได้แม่นยำขึ้น",
        "manual_form1_heading": "ประเภทสินค้า", "manual_form1_note": "เลือกประเภทที่ใกล้เคียงที่สุดก่อน ไม่จำเป็นต้องตรงเป๊ะ",
        "manual_form2_heading": "ข้อมูลสินค้า", "manual_form2_note": "แค่ชื่อสินค้าและราคาก็เช็กได้ ถ้าใส่คำอธิบายด้วยจะได้คำแนะนำการซื้อที่ดีขึ้น",
        "manual_title_label": "ชื่อสินค้า", "manual_price_label": "ราคาขาย", "manual_help_language_label": "ภาษาของคำแนะนำ",
        "manual_desc_label": "คำอธิบายจากผู้ขาย",
        "manual_desc_placeholder": "เช่น ระยะเวลาที่ใช้งาน อุปกรณ์ที่แถม ตำหนิ (ถ้ามี) สถานที่นัดรับ ฯลฯ",
        "manual_option_row_title": "ตัวเลือกเพิ่มเติม",
        "manual_checkbox_use_ai": "รับคำแนะนำการซื้อ",
        "manual_submit": "เช็กราคา",
        "notice_need_title_title": "กรุณากรอกชื่อสินค้า",
        "notice_need_title_copy": "เช่น iPhone 13, โต๊ะ, จักรยาน — เขียนให้ชัดเจนจะได้ผลลัพธ์ที่แม่นยำขึ้น",
        "section_price_analysis": "ผลการวิเคราะห์ราคา",
        "price_disclaimer": "ราคาที่คาดการณ์คำนวณจากข้อมูลการซื้อขายที่แอปเก็บรวบรวมไว้ ก่อนซื้อขายจริงควรตรวจสอบสภาพสินค้าและอุปกรณ์ที่แถมด้วย",
        "dashboard_hero_title": "ดูราคาสินค้ามือสองในปูซาน",
        "dashboard_hero_subtitle": "ดูข้อมูลสินค้าที่เก็บรวบรวมแยกตามหมวดหมู่ หรือค้นหาสินค้าที่สนใจได้ทันที",
        "dashboard_no_data": "ไม่พบไฟล์ used_items_cleaned.csv",
        "section_overview": "ภาพรวม",
        "overview_total_items": "สินค้าที่ลงทะเบียน", "overview_avg_price": "ราคาเฉลี่ย", "overview_median_price": "ราคามัธยฐาน", "overview_category_count": "จำนวนประเภทสินค้า",
        "section_category_price": "ราคาตามหมวดหมู่",
        "category_stat_template": "{count:,} รายการ · เฉลี่ย {avg}",
        "section_browse_items": "ค้นหาสินค้า",
        "filter_category_label": "หมวดหมู่", "filter_keyword_label": "คำค้นหา", "filter_sort_label": "การจัดเรียง",
        "sort_price_asc": "ราคาต่ำไปสูง", "sort_price_desc": "ราคาสูงไปต่ำ", "sort_recent": "เก็บข้อมูลล่าสุด",
        "all_option": "ทั้งหมด",
        "item_count_template": "{count:,} รายการ",
        "view_post_link": "ดูประกาศ",
        "expander_full_table": "ดูข้อมูลทั้งหมดแบบตาราง",
        "col_town": "พื้นที่ที่เก็บข้อมูล", "col_title": "ชื่อเรื่อง", "col_price": "ราคา", "col_category": "หมวดหมู่", "col_link": "ลิงก์ประกาศ",
        "translate_hero_title": "แปลงข้อความที่จะส่งให้ผู้ขายเป็นภาษาเกาหลี",
        "translate_hero_subtitle": "ไม่ต้องเช็กราคาก่อนก็แปลข้อความได้ทุกเมื่อ แล้วคัดลอกไปใช้ได้เลย",
        "translate_source_lang_label": "ภาษาของฉัน",
        "translate_input_label": "ข้อความที่ฉันเขียน",
        "translate_input_placeholder": "เช่น สินค้ายังอยู่ไหมคะ/ครับ พรุ่งนี้นัดรับใกล้โรงเรียนได้ไหม?",
        "translate_button": "แปลเป็นภาษาเกาหลี",
        "translating_spinner": "กำลังแปล...",
        "translate_empty_message": "กรุณากรอกข้อความที่ต้องการแปล",
        "openai_key_missing_help": "ไม่ได้ตั้งค่า OpenAI API Key จึงสร้างคำแนะนำการซื้อไม่ได้",
        "openai_pkg_missing_help": "ไม่ได้ติดตั้งแพ็กเกจ openai จึงสร้างคำแนะนำการซื้อไม่ได้",
        "openai_key_missing_translate": "ไม่ได้ตั้งค่า OpenAI API Key จึงแปลไม่ได้",
        "openai_pkg_missing_translate": "ไม่ได้ติดตั้งแพ็กเกจ openai จึงแปลไม่ได้",
    },
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
            background: #ffffff;
            color: var(--ink);
            border: 1px solid #edf0f2;
            box-shadow: 0 8px 18px rgba(31, 41, 51, 0.05);
            padding: 0.7rem 0.75rem;
            margin-bottom: 10px;
        }

        [data-testid="stSidebar"] .st-key-home_logo button {
            width: 52px;
            height: 52px;
            border-radius: 18px;
            background: var(--karrot);
            color: #ffffff;
            border: 0;
            box-shadow: 0 14px 28px rgba(255, 111, 15, 0.28);
            font-size: 1.45rem;
            padding: 0;
            margin-bottom: 14px;
        }

        [data-testid="stSidebar"] .st-key-home_logo button:hover {
            background: var(--karrot-dark);
            color: #ffffff;
            transform: translateY(-1px);
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


def render_hero(title, subtitle, kicker=None):
    kicker = kicker if kicker is not None else t("app_kicker")
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
    slug = STATUS_KEY_MAP.get(status, "unknown")
    return t(f"status_title_{slug}"), t(f"status_copy_{slug}")


def render_price_summary(price_result):
    status = price_result.get("가격판단", "판단 불가")
    slug = STATUS_KEY_MAP.get(status, "unknown")
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
            <div class="result-kicker">{escape(t("price_check_result"))}</div>
            <div class="result-badge" style="background:{color};">{escape(t(f"status_badge_{slug}"))}</div>
            <div class="result-title">{escape(title)}</div>
            <div class="result-copy">{escape(copy)}</div>
            <div class="result-numbers">
                <div class="result-number">
                    <div class="result-number-label">{escape(t("actual_price"))}</div>
                    <div class="result-number-value">{escape(format_won(price_result.get("실제가격")))}</div>
                </div>
                <div class="result-number">
                    <div class="result-number-label">{escape(t("expected_price"))}</div>
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
    st.session_state["reset_version"] = st.session_state.get("reset_version", 0) + 1
    for key in ("analysis_result", "manual_result", "translate_page_result"):
        st.session_state.pop(key, None)


def widget_key(name):
    return f"{name}_{st.session_state.get('reset_version', 0)}"


def current_lang_code():
    return LANG_CODE_MAP.get(st.session_state.get("ui_lang", "한국어"), "ko")


def t(key):
    code = current_lang_code()
    value = TRANSLATIONS.get(code, {}).get(key)
    if value is not None:
        return value
    return TRANSLATIONS["ko"].get(key, key)


def category_label(category):
    code = current_lang_code()
    value = CATEGORY_LABELS_I18N.get(code, {}).get(category)
    if value is not None:
        return value
    return CATEGORY_LABELS_I18N["ko"].get(category, category)


def init_ui_lang():
    if "ui_lang" not in st.session_state:
        lang_code = st.query_params.get("lang", "ko")
        st.session_state["ui_lang"] = CODE_TO_LANG_LABEL.get(lang_code, "한국어")


def render_language_selector():
    st.sidebar.selectbox(t("ui_language_label"), list(LANGUAGE_OPTIONS.keys()), key="ui_lang")
    st.query_params["lang"] = LANG_CODE_MAP[st.session_state["ui_lang"]]


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
        return t("openai_key_missing_help")

    try:
        from openai import OpenAI
    except Exception:
        return t("openai_pkg_missing_help")

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


def build_translate_prompt(text, source_language):
    return f"""
너는 부산외국어대학교 유학생을 위한 중고거래 번역 도우미야.
아래 메시지는 유학생이 당근마켓 판매자에게 보내고 싶어서 {source_language}로 작성한 문장이야.
이 메시지를 자연스러운 한국어 존댓말 채팅 문장으로 번역해줘.

[원문 - {source_language}]
{text}

번역할 때 지킬 것:
- 실제 채팅에 그대로 복사해서 보낼 수 있는 자연스러운 한국어 문장으로 번역해.
- 너무 딱딱한 번역투가 아니라 실제 사람이 쓰는 채팅처럼 자연스럽게 써.
- 원문에 없는 내용을 추가하거나 설명을 덧붙이지 마.
- 번역된 한국어 문장만 출력하고 다른 말은 하지 마.
""".strip()


def translate_message_to_korean(text, source_language):
    if not text or not text.strip():
        return t("translate_empty_message")

    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        return t("openai_key_missing_translate")

    try:
        from openai import OpenAI
    except Exception:
        return t("openai_pkg_missing_translate")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=get_secret("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You translate chat messages from international students into natural, polite Korean for secondhand marketplace sellers."},
            {"role": "user", "content": build_translate_prompt(text.strip(), source_language)},
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


def render_translate_page():
    render_hero(t("translate_hero_title"), t("translate_hero_subtitle"))

    labels = list(LANGUAGE_OPTIONS.keys())
    default_label = st.session_state.get("ui_lang", "한국어")
    default_index = labels.index(default_label) if default_label in labels else labels.index("한국어")
    source_label = st.selectbox(
        t("translate_source_lang_label"),
        labels,
        index=default_index,
        key=widget_key("translate_source_lang"),
    )

    user_message = st.text_area(
        t("translate_input_label"),
        placeholder=t("translate_input_placeholder"),
        key=widget_key("translate_page_input"),
    )

    if st.button(t("translate_button"), type="primary", key=widget_key("translate_page_btn")):
        with st.spinner(t("translating_spinner")):
            translated = translate_message_to_korean(user_message, LANGUAGE_OPTIONS[source_label])
        st.session_state["translate_page_result"] = translated

    translated_message = st.session_state.get("translate_page_result")
    if translated_message:
        st.code(translated_message, language=None)


def render_analysis_page():
    render_hero(t("link_hero_title"), t("link_hero_subtitle"))

    st.markdown(
        f"""
        <div class="step-strip">
            <div class="step-card">
                <div class="step-num">1</div>
                <div class="step-title">{escape(t("link_step1_title"))}</div>
                <div class="step-copy">{escape(t("link_step1_copy"))}</div>
            </div>
            <div class="step-card">
                <div class="step-num">2</div>
                <div class="step-title">{escape(t("link_step2_title"))}</div>
                <div class="step-copy">{escape(t("link_step2_copy"))}</div>
            </div>
            <div class="step-card">
                <div class="step-num">3</div>
                <div class="step-title">{escape(t("link_step3_title"))}</div>
                <div class="step-copy">{escape(t("link_step3_copy"))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("analysis_form"):
        st.markdown(
            f"""
            <div class="form-heading"><span>1</span>{escape(t("link_form1_heading"))}</div>
            <div class="form-note">{escape(t("link_form1_note"))}</div>
            """,
            unsafe_allow_html=True,
        )
        url = st.text_input(
            t("link_form1_heading"),
            placeholder="https://www.daangn.com/kr/buy-sell/...",
            label_visibility="collapsed",
            key=widget_key("link_url"),
        )

        st.markdown(
            f"""
            <div class="form-heading"><span>2</span>{escape(t("link_form2_heading"))}</div>
            <div class="form-note">{escape(t("link_form2_note"))}</div>
            """,
            unsafe_allow_html=True,
        )
        selected_category = st.pills(
            t("link_form2_heading"),
            CATEGORY_OPTIONS,
            default="전자기기",
            format_func=lambda value: category_label(value),
            key=widget_key("link_category"),
            label_visibility="collapsed",
            width="stretch",
        )
        selected_category = selected_category or "기타"

        st.markdown(f'<div class="option-row-title">{escape(t("option_row_title"))}</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns([1.15, 1.2, 1.2, 1.15])
        with col1:
            use_manual_price = st.checkbox(t("link_checkbox_manual_price"), value=True, key=widget_key("link_manual_price_enabled"))
        with col2:
            manual_price = st.number_input(t("price_label"), min_value=0, step=1000, value=0, disabled=not use_manual_price, key=widget_key("link_manual_price"))
        with col3:
            language_label = st.selectbox(t("language_select_label"), list(LANGUAGE_OPTIONS.keys()), key=widget_key("link_language"))
        with col4:
            use_ai = st.checkbox(t("link_checkbox_use_ai"), value=True, key=widget_key("link_use_ai"))

        submitted = st.form_submit_button(t("link_submit"), type="primary", use_container_width=True)

    if submitted:
        if not url.strip():
            render_notice(t("notice_need_link_title"), t("notice_need_link_copy"))
            st.session_state.pop("analysis_result", None)
        else:
            selected_price = int(manual_price) if use_manual_price else None
            with st.spinner(t("link_spinner_analyzing")):
                try:
                    result = analyze_used_item_link(
                        url=url.strip(),
                        language=LANGUAGE_OPTIONS[language_label],
                        manual_price=selected_price,
                        use_ai=use_ai,
                        category=selected_category,
                    )
                except Exception as exc:
                    render_notice(t("notice_link_error_title"), t("notice_link_error_copy"))
                    st.caption(f"{t('error_info_prefix')}{exc}")
                    st.session_state.pop("analysis_result", None)
                    result = None

            if result is not None:
                st.session_state["analysis_result"] = result

    result = st.session_state.get("analysis_result")
    if not result:
        return

    item_info = result["상품정보"]
    price_result = result["가격판단결과"]

    render_price_summary(price_result)

    st.markdown(f'<div class="section-title">{escape(t("section_item_info"))}</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_info_card(t("info_title"), item_info["제목"])
    with c2:
        render_info_card(t("info_price"), item_info["가격"])
    with c3:
        render_info_card(t("info_category"), category_label(price_result["카테고리"]))
    with c4:
        render_info_card(t("info_expected_price"), format_won(price_result["예측적정가격"]))

    with st.expander(t("expander_seller_desc"), expanded=True):
        st.write(item_info["상세설명"])

    if result["AI설명"]:
        st.markdown(f'<div class="section-title">{escape(t("section_purchase_help"))}</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(result["AI설명"])


def render_manual_page():
    render_hero(t("manual_hero_title"), t("manual_hero_subtitle"))

    with st.form("manual_form"):
        st.markdown(
            f"""
            <div class="form-heading"><span>1</span>{escape(t("manual_form1_heading"))}</div>
            <div class="form-note">{escape(t("manual_form1_note"))}</div>
            """,
            unsafe_allow_html=True,
        )
        selected_category = st.pills(
            t("manual_form1_heading"),
            CATEGORY_OPTIONS,
            default="전자기기",
            format_func=lambda value: category_label(value),
            key=widget_key("manual_category"),
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
            f"""
            <div class="form-heading"><span>2</span>{escape(t("manual_form2_heading"))}</div>
            <div class="form-note">{escape(t("manual_form2_note"))}</div>
            """,
            unsafe_allow_html=True,
        )
        title = st.text_input(
            t("manual_title_label"),
            placeholder=CATEGORY_PLACEHOLDERS.get(selected_category, CATEGORY_PLACEHOLDERS["기타"]),
            key=widget_key("manual_title"),
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            price = st.number_input(t("manual_price_label"), min_value=0, step=1000, value=10000, key=widget_key("manual_price"))
        with col2:
            language_label = st.selectbox(t("manual_help_language_label"), list(LANGUAGE_OPTIONS.keys()), key=widget_key("manual_language"))
        description = st.text_area(
            t("manual_desc_label"),
            placeholder=t("manual_desc_placeholder"),
            key=widget_key("manual_description"),
        )

        st.markdown(f'<div class="option-row-title">{escape(t("manual_option_row_title"))}</div>', unsafe_allow_html=True)
        use_ai = st.checkbox(t("manual_checkbox_use_ai"), value=True, key=widget_key("manual_ai"))
        submitted = st.form_submit_button(t("manual_submit"), type="primary", use_container_width=True)

    if submitted:
        if not title.strip():
            render_notice(t("notice_need_title_title"), t("notice_need_title_copy"))
            st.session_state.pop("manual_result", None)
        else:
            item_info = {
                "제목": title.strip(),
                "카테고리": selected_category,
                "가격": format_won(price),
                "가격_numeric": int(price),
                "상세설명": description.strip() or "상세설명 없음",
                "링크": "",
            }
            price_result = predict_price_from_item(item_info, manual_price=int(price))

            ai_text = None
            if use_ai:
                with st.spinner(t("manual_spinner_generating")):
                    ai_text = generate_multilingual_result(item_info, price_result, LANGUAGE_OPTIONS[language_label])

            st.session_state["manual_result"] = {
                "상품정보": item_info,
                "가격판단결과": price_result,
                "AI설명": ai_text,
            }

    result = st.session_state.get("manual_result")
    if not result:
        return

    item_info = result["상품정보"]
    price_result = result["가격판단결과"]

    render_price_summary(price_result)

    st.markdown(f'<div class="section-title">{escape(t("section_price_analysis"))}</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        render_info_card(t("info_category"), category_label(price_result["카테고리"]))
    with col2:
        render_info_card(t("info_price"), format_won(price_result["실제가격"]))
    with col3:
        render_info_card(t("info_expected_price"), format_won(price_result["예측적정가격"]))

    st.markdown(
        f"""
        <div class="small-muted">
        {escape(t("price_disclaimer"))}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result["AI설명"]:
        st.markdown(f'<div class="section-title">{escape(t("section_purchase_help"))}</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(result["AI설명"])


def render_dashboard_page():
    render_hero(
        t("dashboard_hero_title"),
        t("dashboard_hero_subtitle"),
        kicker=t("nav_dashboard"),
    )
    df = load_data()
    if df.empty:
        st.warning(t("dashboard_no_data"))
        return

    df = df.copy()
    df["가격_numeric"] = pd.to_numeric(df["가격_numeric"], errors="coerce")
    df = df.dropna(subset=["가격_numeric"])

    st.markdown(f'<div class="section-title">{escape(t("section_overview"))}</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_info_card(t("overview_total_items"), t("item_count_template").format(count=len(df)))
    with col2:
        render_info_card(t("overview_avg_price"), format_won(df["가격_numeric"].mean()))
    with col3:
        render_info_card(t("overview_median_price"), format_won(df["가격_numeric"].median()))
    with col4:
        render_info_card(t("overview_category_count"), t("item_count_template").format(count=df["카테고리"].nunique()))

    category_summary = (
        df.groupby("카테고리", as_index=False)
        .agg(상품수=("제목", "count"), 평균가격=("가격_numeric", "mean"), 중앙값=("가격_numeric", "median"))
        .sort_values("상품수", ascending=False)
    )

    st.markdown(f'<div class="section-title">{escape(t("section_category_price"))}</div>', unsafe_allow_html=True)
    category_cols = st.columns(3)
    for index, row in category_summary.head(6).reset_index(drop=True).iterrows():
        with category_cols[index % 3]:
            st.markdown(
                f"""
                <div class="category-stat">
                    <div class="category-name">{escape(category_label(row['카테고리']))}</div>
                    <div class="category-price">{escape(format_won(row['중앙값']))}</div>
                    <div class="category-count">{escape(t("category_stat_template").format(count=int(row['상품수']), avg=format_won(row['평균가격'])))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(f'<div class="section-title">{escape(t("section_browse_items"))}</div>', unsafe_allow_html=True)
    filter_col1, filter_col2, filter_col3 = st.columns([1.2, 2, 1.2])
    with filter_col1:
        selected_category = st.selectbox(
            t("filter_category_label"),
            ["전체"] + category_summary["카테고리"].tolist(),
            format_func=lambda value: t("all_option") if value == "전체" else category_label(value),
            key=widget_key("dashboard_category"),
        )
    with filter_col2:
        keyword = st.text_input(t("filter_keyword_label"), placeholder="아이폰, 책상, 자전거...", key=widget_key("dashboard_keyword"))
    with filter_col3:
        sort_options = ["낮은 가격순", "높은 가격순", "최근 수집순"]
        sort_label_map = {"낮은 가격순": t("sort_price_asc"), "높은 가격순": t("sort_price_desc"), "최근 수집순": t("sort_recent")}
        sort_label = st.selectbox(
            t("filter_sort_label"),
            sort_options,
            format_func=lambda value: sort_label_map[value],
            key=widget_key("dashboard_sort"),
        )

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

    st.caption(t("item_count_template").format(count=len(filtered)))
    for _, row in sorted_items.head(12).iterrows():
        title = escape(str(row.get("제목", "")))
        category = escape(category_label(row.get("카테고리", "")))
        town = escape(str(row.get("수집동네", "")))
        price = escape(str(row.get("가격", format_won(row.get("가격_numeric")))))
        link = escape(str(row.get("게시글링크", "")))
        link_html = f' · <a href="{link}" target="_blank">{escape(t("view_post_link"))}</a>' if link.startswith("http") else ""
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

    with st.expander(t("expander_full_table")):
        display_df = filtered[["수집동네", "제목", "가격", "카테고리", "게시글링크"]].head(200).rename(
            columns={
                "수집동네": t("col_town"),
                "제목": t("col_title"),
                "가격": t("col_price"),
                "카테고리": t("col_category"),
                "게시글링크": t("col_link"),
            }
        )
        st.dataframe(display_df, use_container_width=True)


def main():
    load_dotenv_if_exists()
    inject_theme_style()
    init_ui_lang()

    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "링크 분석"
    if "reset_version" not in st.session_state:
        st.session_state["reset_version"] = 0

    if st.sidebar.button("🥕", help=t("home_help"), key="home_logo", use_container_width=False):
        reset_inputs()
        st.session_state["nav_page"] = "링크 분석"
        st.rerun()

    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-title">{t("brand_title_html")}</div>
            <div class="sidebar-copy">{escape(t("brand_copy"))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_language_selector()

    if st.sidebar.button(t("refresh_button"), use_container_width=True):
        reset_inputs()
        st.rerun()

    st.sidebar.markdown(f'<div class="sidebar-label">{escape(t("menu_label"))}</div>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        t("menu_label"),
        ["링크 분석", "직접 입력", "데이터 대시보드", "메시지 번역"],
        format_func=lambda value: t(NAV_KEY_TO_TKEY[value]),
        label_visibility="collapsed",
        key="nav_page",
    )

    model = load_model()
    if model is None:
        st.sidebar.markdown(
            f"""
            <div class="sidebar-status warn">
                <span class="sidebar-status-dot"></span>
                {escape(t("status_model_missing"))}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            f"""
            <div class="sidebar-status ok">
                <span class="sidebar-status-dot"></span>
                {escape(t("status_model_ready"))}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not get_secret("OPENAI_API_KEY"):
        st.sidebar.markdown(
            f"""
            <div class="sidebar-status muted">
                <span class="sidebar-status-dot"></span>
                {escape(t("status_ai_key_missing"))}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if page == "링크 분석":
        render_analysis_page()
    elif page == "직접 입력":
        render_manual_page()
    elif page == "데이터 대시보드":
        render_dashboard_page()
    else:
        render_translate_page()


if __name__ == "__main__":
    main()

