from transformers import GenerationConfig


generation_config = GenerationConfig(
    # do_sample=False,
    temperature=0.1,
    top_p=0.95,
    seed=42,
)

def make_prompt(target_sent, location, sent1, sent2, sent3, tokenizer, flag_vlm=False):
    instruction = (
        "[지침]\n[대상발화]의 선행문 또는 후행문으로 가장 자연스럽고 적절한 발화를 제시된 후보 발화 중에서 선택하시오. [생성 기준]을 꼼꼼히 읽고 이해하는 것이 중요합니다.\n\n"
        "[생성 기준]\n"
        "1 - 당신은 [대상발화]의 [발화위치]에 가장 적합한 발화를 후보발화 중에서 선택하는 챗봇입니다. '발화1', '발화2', '발화3' 중에서 1개를 생성하시오.\n"
        "2 - [발화위치]는 '선행문', '후행문'입니다. '선행문'은 [대상발화]보다 선행하는 문장이고, '후행문'은 [대상발화]보다 후행하는 문장입니다.\n"
        "3 - 출력은 '발화1', '발화2', '발화3' 중에서 1개만 생성하시오."
    )
    sentence = f"[대상발화]\n{target_sent}\n\n[발화위치]\n{location}"
    candidates = f"[후보 발화]\n발화1:{sent1}\n발화2:{sent2}\n발화3:{sent3}"
    user_prompt = instruction + "\n\n" + sentence + "\n\n" + candidates

    # LLM
    if not flag_vlm:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ]
    # Vision-Language 
    else:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            }
        ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False, # Qwen3, Switches between thinking and non-thinking modes. Default is True.
        return_tensors="pt",
        # return_dict=True, # tokenize=True인 경우에 사용
    )

    return prompt
