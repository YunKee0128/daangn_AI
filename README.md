# 유학생을 위한 중고거래 다국어 도우미

## 프로젝트 소개
이 프로젝트는 유학생이 한국 중고거래 게시글을 더 쉽게 이해할 수 있도록 돕는 AI 서비스이다.
사용자가 상품 제목, 판매 가격, 상세설명을 입력하면 머신러닝 모델이 수집 데이터 기준 참고 가격을 예측하고, LLM이 영어, 일본어, 중국어로 상품 설명과 거래 주의사항을 제공한다.

## 주요 기능
- 중고거래 상품 가격 참고 예측
- 실제 판매 가격과 예측 참고 가격 비교
- 저렴 / 적정 / 비쌈 판단
- 영어, 일본어, 중국어 다국어 설명 생성
- 구매 전 확인사항 제공
- 판매자에게 보낼 수 있는 메시지 생성

## 사용 기술
- Python
- Streamlit
- pandas
- scikit-learn
- RandomForestRegressor
- LangChain
- OpenAI API

## 프로젝트 파일 구조
- app.py : Streamlit 앱 실행 파일
- used_item_price_model.pkl : 학습된 머신러닝 가격 예측 모델
- used_items_cleaned.csv : 전처리 완료 데이터
- requirements.txt : 실행에 필요한 패키지 목록
- README.md : 프로젝트 설명
- .gitignore : GitHub에 올리지 않을 파일 설정

Streamlit Cloud에서 앱이 정상 실행되려면 `app.py`, `used_item_price_model.pkl`, `used_items_cleaned.csv`, `requirements.txt`, `README.md`, `.gitignore` 파일을 GitHub 저장소에 포함해야 한다.

## 실행 방법
터미널에서 다음 명령어를 실행한다.

```bash
streamlit run app.py
```

## API Key 설정
OpenAI API Key는 `.env` 파일 또는 Streamlit Cloud Secrets에 등록해야 한다.
GitHub에는 API Key가 포함된 `.env` 파일을 절대 올리면 안 된다.

로컬 실행 시 `.env` 예시는 다음과 같다.

```env
OPENAI_API_KEY=본인의_API_KEY
```

Streamlit Cloud 배포 시에는 앱 설정의 Secrets에 다음과 같이 등록한다.

```toml
OPENAI_API_KEY = "본인의_API_KEY"
```

## 주의사항
예측 가격은 수집 데이터 기준 참고 가격이며 실제 중고거래 시세와 다를 수 있다.
학습 데이터에 특정 고가 상품군이 부족한 경우 예측 가격이 실제 시세보다 낮거나 높게 나올 수 있다.
따라서 본 서비스의 가격 판단 결과는 절대적인 시세가 아니라 유학생을 위한 참고 정보로 활용한다.
