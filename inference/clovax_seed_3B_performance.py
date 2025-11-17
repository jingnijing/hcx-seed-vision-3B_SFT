''' 감정 이미지 + 텍스트를 이용한 복잡한 JSON 구조 테스트 '''

import time
import json
import torch
import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# 모델 로딩
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 테스트 데이터 로드 (영어 입력 문장 + 이미지 경로 포함 CSV)
csv_path = "image_emotion_data.csv"  # 반드시 '영어 입력 문장', '이미지 경로' 컬럼 필요
df = pd.read_csv(csv_path)

results = []

for idx, row in df.iterrows():
    sentence = row["한글 입력 문장"]
    image_path = row["이미지 경로"]

    print(f"\n=== 테스트 {idx + 1} ===")

    chat = [
        {
                "role": "system",
                "content": {
                "type": "text",
                "text": (
                        "너는 텍스트와 이미지를 각각 독립적으로 분석하고, 종합적으로 감정 및 관찰 정보를 JSON 형식으로 정확히 반환하는 한국어 감정 분석 전문가야.\n"
                        "다음 규칙을 반드시 지켜야 한다:\n"
                        "- 텍스트 분석 결과와 이미지 분석 결과를 반드시 분리해서 작성\n"
                        "- 종합 감정(final_emotion)은 두 결과를 바탕으로 최종 판단\n"
                        "- 'emotion_text', 'emotion_image', 'final_emotion' 값은 반드시 '기쁨', '슬픔', '놀람', '분노', '불안', '중립', 'unknown' 중 하나의 한국어 단어만 작성하라\n"
                        "- 'observable_signs.summary'는 한국어 문장으로 작성하라\n"
                        "- 'observable_signs.details'는 반드시 아래 구조의 리스트로 작성하라:\n"
                        "  [\n"
                        "    {\"type\": \"표정\", \"description\": \"눈물이 고인 눈빛\", \"confidence\": 0.87},\n"
                        "    {\"type\": \"제스처\", \"description\": \"고개를 숙인 자세\", \"confidence\": 0.92}\n"
                        "  ]\n"
                        "- 'intensity'는 0.0 ~ 1.0 사이의 실수로 작성하라\n"
                        "- 'additional_info' 필드는 반드시 포함하며 다음 정보를 정확히 포함해야 한다:\n"
                        "  - 'source_language': 원본 문장의 언어 (예: \"영어\")\n"
                        "  - 'target_language': 번역된 언어 (예: \"한국어\")\n"
                        "  - 'translation_quality': \"높음\", \"중간\", \"낮음\" 중 하나\n"
                        "  - 'timestamp': ISO 8601 형식의 날짜/시간 문자열 (예: \"2025-06-30T12:34:56Z\")\n"
                        "  - 'reviewed_by': 다음 구조의 리스트를 포함\n"
                        "    [\n"
                        "      {\"reviewer_id\": \"AI-001\", \"status\": \"확인 완료\"},\n"
                        "      {\"reviewer_id\": \"AI-002\", \"status\": \"자동 검증\"}\n"
                        "    ]\n"
                        "- 반드시 아래 JSON 구조를 엄격히 준수해라. 내용은 반드시 새롭게 분석해 작성하고, 형식을 절대 위반하지 마라.\n"
                        "- 각 필드는 반드시 한국어로 실질적이고 구체적인 내용을 작성해야 한다.\n"
                        "- 빈 값으로 남기거나 의미 없는 출력은 절대 금지다.\n"

                )
                }
        },
        {
                "role": "user",
                "content": {
                "type": "text",
                "text": (
                        f"다음 문장과 이미지를 분석해 감정을 판단해라.\n"
                        f"문장:\n\"{sentence}\""
                )
                }
        },
        {
                "role": "user",
                "content": {
                "type": "image",
                "filename": os.path.basename(image_path),
                "image": image_path
                }
        },
        {
                "role": "user",
                "content": {
                "type": "text",
                "text": (
                "반드시 아래 JSON 구조로만 결과를 출력하라. 내용은 새롭게 분석해 작성하고, 형식을 절대 위반하지 마라.\n"
                "{\n"
                "  \"emotion_ko\": \"\",\n"
                "  \"observable_signs\": {\n"
                "    \"summary\": \"\",\n"
                "    \"details\": [\n"
                "      {\"type\": \"\", \"description\": \"\", \"confidence\": 0.0}\n"
                "    ]\n"
                "  },\n"
                "  \"intensity\": 0.0,\n"
                "  \"additional_info\": {\n"
                "    \"source_language\": \"\",\n"
                "    \"target_language\": \"\",\n"
                "    \"translation_quality\": \"\",\n"
                "    \"timestamp\": \"\",\n"
                "    \"reviewed_by\": [\n"
                "      {\"reviewer_id\": \"\", \"status\": \"\"}\n"
                "    ]\n"
                "  }\n"
                "}\n"
                "위 구조를 정확히 지키되, 모든 필드에 대해 한국어로 실제 분석한 내용을 반드시 작성하라. 빈 값이나 의미 없는 값은 절대 허용되지 않는다."
                )
                }
        }
        ]

    # 멀티모달 전처리
    new_chat, all_images, is_video_list = preprocessor.load_images_videos(chat)
    print(f"\n=== 이미지 로딩 개수: {len(all_images)} ===")
    
    final_prompt = tokenizer.apply_chat_template(new_chat, tokenize=False)
    print("\n=== 최종 모델 입력 프롬프트 ===\n")
    print(final_prompt)
    
    kwargs = {}
    if all_images:
        kwargs = preprocessor(all_images, is_video_list=is_video_list)

    input_ids = tokenizer.apply_chat_template(
        new_chat, return_tensors="pt", tokenize=True, add_generation_prompt=True
    ).to(device)

    start = time.time()
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.8,
        temperature=0.8,
        repetition_penalty=1.0,
        **kwargs
    )
    elapsed = (time.time() - start) * 1000

    output_text = tokenizer.batch_decode(output_ids)[0]

    # "assistant" 태그 이후 JSON만 추출
    json_candidate = output_text.split("<|im_start|>assistant")[-1]
    json_start = json_candidate.find("{")
    json_end = json_candidate.rfind("}") + 1
    json_str = json_candidate[json_start:json_end]

    print("출력 결과:\n", json_str)
    print(f"지연시간: {elapsed:.2f}ms")

    syntax_valid = False
    try:
        parsed_json = json.loads(json_str)
        syntax_valid = True
    except Exception as e:
        print(f"JSON 파싱 실패: {e}")

    print(f"JSON 구문 유효성: {syntax_valid}")

    results.append({
        "ID": row["ID"],
        "입력 문장": sentence,
        "이미지 경로": image_path,
        "모델 출력": json_str,
        "지연시간(ms)": round(elapsed, 2),
        "JSON 구문 유효성": syntax_valid
    })

# 결과 저장
results_df = pd.DataFrame(results)
results_df.to_csv("clova_test_results_text_image_json_3.csv", index=False)
print("\n전체 테스트 완료, 결과 파일: clova_test_results_text_image_json_3.csv")



''' 지연시간(번역 ~ JSON 파싱까지) 평균 구하기 '''
# import pandas as pd

# # 테스트 결과 CSV 경로
# csv_path = "clova_test_results_text_image_json_2.csv"

# # 결과 파일 로드
# df = pd.read_csv(csv_path)

# # 지연시간 컬럼 평균 계산
# avg_latency = df["지연시간(ms)"].mean()

# print(f"전체 평균 지연시간: {avg_latency:.2f} ms")

