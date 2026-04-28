import json
import os
from PIL import Image

import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('omw-1.4', quiet=True)

import evaluate
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from loguru import logger as eval_logger

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")

    
MM_VET_PROMPT = """Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. <AND> in the ground truth means it is totally right only when all elements in the ground truth are present in the prediction, and <OR> means it is totally right when any one element in the ground truth is present in the prediction. The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Just complete the last space of the correctness score.
gpt_query_prompt | Ground truth | Prediction | Correctness
--- | --- | --- | ---
What is x in the equation? | -1 <AND> -5 | x = 3 | 0.0
What is x in the equation? | -1 <AND> -5 | x = -1 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 or 5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -1 or x = -5 | 1.0
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green. | 0.4
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0
"""

BASE_PROMPT = """You are an expert guide for visually impaired individuals. Your task is to provide a concise explanation based on the following guidelines, delivering the content as if speaking naturally without section breaks.

Guidelines:
1) Surroundings and Position: Summarize where the person is, the general environment, their current position, and any nearby landmarks in 1-2 sentences.
2) Hazards:
   - For each direction (10, 11, 12, 1, and 2 o'clock), combine all hazards in that direction into exactly one sentence, mentioning approximate distance(s) and reason(s) they are dangerous.
   - Follow the order of 10, 11, 12, 1, and 2 o'clock.
3) Navigation: After describing all hazards, provide a single, concise sentence on how to safely navigate or avoid them overall.
"""


# 평가 지표 로드
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")


def get_object_info(object_info, is_ordering_clock=False, is_include_dangerous=False, is_only_danger=False):
    """객체 정보 문자열 생성"""
    if is_ordering_clock:
        text_dict = {str(i): [] for i in range(1, 13)}  # 시계 방향별 텍스트 저장
    else:
        text = ""
    
    for i, cls in enumerate(object_info['objects']):
        if is_only_danger and object_info['is_dangerous'][i].lower().strip() != "yes":
            continue
            
        # 객체 정보 텍스트 생성
        obj_text = f"Object: {cls}\n"
        obj_text += f"- Bounding Box (x1,y1,x2,y2): [{', '.join(map(str, map(int, object_info['bboxes'][i])))}]\n"
        obj_text += f"- Bounding Box Ratio (x1,y1,x2,y2): [{', '.join(f'{x:.2f}' for x in object_info['ratio_bboxes'][i])}]\n"
        obj_text += f"- Depth: {object_info['depths'][i]:.2f}m\n"
        obj_text += f"- Direction: {object_info['directions'][i]} o'clock\n"
        
        if is_include_dangerous:
            obj_text += f"- Is Dangerous: {object_info['is_dangerous'][i]}\n"
            if object_info['is_dangerous'][i] and object_info['is_dangerous'][i].lower().strip() == "yes":
                obj_text += f"- Why Dangerous: {object_info['why_dangerous'][i]}\n"
            else:
                obj_text += f"- Why Not Dangerous: {object_info['why_dangerous'][i]}\n"
        obj_text += "\n"
        
        if is_ordering_clock:
            direction = str(object_info['directions'][i])
            if direction in text_dict:
                text_dict[direction].append(obj_text)
        else:
            text += obj_text
    
    if is_ordering_clock:
        text = ""
        # 10시부터 2시까지 순서대로 텍스트 조합
        for hour in ['10', '11', '12', '1', '2']:
            text += ''.join(text_dict[hour])
            
    return text.strip()

def convert_distance_to_steps(object_info, steps_ratio=0.7):
    object_info_text_list = []
    
    for line in object_info.split("\n"):
        if "Depth" in line:
            depth = float(line.split(":")[1].split("m")[0].strip())
            steps = depth / steps_ratio # 1스텝이 0.7미터이므로 미터를 0.7로 나눔
            object_info_text_list.append(f"- Depth: {depth}m ({round(steps)} steps)")
        else:
            object_info_text_list.append(line)
            
    return "\n".join(object_info_text_list).strip()

def get_fewshot_prompt(example_num):
    example_list = [
        "You're on a bustling city street with buildings on your left, and the sidewalk and storefronts on your right. At 10 o'clock, about five steps away, there's a moving car which is potentially dangerous if you stray off the sidewalk. To navigate safely, stay on the sidewalk, maintaining a safe distance from the road.",
        "You're standing in a lively marketplace with stalls under umbrellas at 12 o'clock and buildings in the background; there are parked vehicles to your sides. At 10 o'clock, approximately 5 steps away, there's a parked car that could obstruct any movement in that direction. At 11 o'clock, there are no immediate hazards. Directly ahead, at 12 o'clock, the market stalls might pose a minor obstacle if you walk too closely. At 1 o'clock, there is a car about 4 steps away, posing a potential obstacle. At 2 o'clock, no significant hazards are present. To navigate safely, proceed slowly towards 12 o'clock while veering slightly to your right to avoid the car at 1 o'clock.",
        "You are in a public plaza with buildings directly ahead at 12 o'clock and an art structure at 2 o'clock, with pedestrians around. At 10 o'clock, there is a barrier post and a pole about 3 steps away which could obstruct your path. At 11 o'clock, trash bins are 4 steps away, which might be a tripping hazard. Directly ahead at 12 o'clock, a foldout sign is 3 steps away, posing a risk of collision. At 2 o'clock, a barrier post is 4 steps away, which could also cause a trip. To safely navigate the area, move slightly to your left and proceed forward, avoiding the central obstacles."
    ]
    return example_list[example_num]


def guidedog_doc_to_visual(doc):
    return [doc["image"]]


def guidedog_socratic_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    fewshot_num_examples = lmms_eval_specific_kwargs['fewshot_num_examples']
    object_info = get_object_info(doc['object_info'], is_ordering_clock=True, is_include_dangerous=False, is_only_danger=True)
    object_info = convert_distance_to_steps(object_info)
    
    text = BASE_PROMPT
    if fewshot_num_examples > 0:
        text += "\n\nExamples:"
        for example_num in range(fewshot_num_examples):
            text += "\nScene Description:"
            text += f"\nObject Info:"
            text += f"\nGuidance:\n{get_fewshot_prompt(example_num)}\n"
    text += "\nRemember to provide a single, flowing explanation without labeled sections, as if talking directly to the visually impaired individual."
    text += "\n\nScene Description: \n[VLM OUTPUT]"
    text += f"\nObject Info: \n{object_info}"
    text += "\nGuidance:"
    return text


def guidedog_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    fewshot_num_examples = lmms_eval_specific_kwargs['fewshot_num_examples']
    
    text = BASE_PROMPT
    if fewshot_num_examples > 0:
        text += "\n\nExamples:"
        for example_num in range(fewshot_num_examples):
            text += f"\n- {get_fewshot_prompt(example_num)}"

    text += "\n\nRemember to provide a single, flowing explanation without labeled sections, as if talking directly to the visually impaired individual."
    return text


def get_tfidf_similarity(predictions, references):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(references + predictions)
    tfidf_similarity = cosine_similarity(tfidf_matrix[:len(references)], tfidf_matrix[len(references):])
    tfidf_scores = np.diag(tfidf_similarity)
    return np.mean(tfidf_scores)


def get_gpt_score(predictions, references):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    scores = []
    question = "Write a single, natural-sounding passage with no section breaks. Begin by providing a 1–2 sentence summary of the person's surroundings and position, including any nearby landmarks. Then, for each direction from 10 to 2 o'clock, combine all hazards in exactly one sentence, specifying approximate distances and why they are dangerous in the order of 10, 11, 12, 1, and 2 o'clock. Finally, offer a single concise sentence explaining how to safely navigate or avoid all hazards."
    for pred, ref in zip(predictions, references):  
        prompt = f"{MM_VET_PROMPT}\n{question} | {ref} | {pred} |"
        response = client.chat.completions.create(
            model="gpt-4-0613",
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        score = response.choices[0].message.content.strip()
        try:
            score = float(score)
        except:
            print(f"Mapping error: {score}")
            score = 0.0
        scores.append(score)
    return np.mean(scores)

                             
def guidedog_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    predictions = result[0]
    references = doc["label"]
    
    bleu_score = bleu.compute(predictions=[predictions], references=[references])
    rouge_score = rouge.compute(predictions=[predictions], references=[references])
    meteor_score = meteor.compute(predictions=[predictions], references=[references])
    bertscore_score = bertscore.compute(predictions=[predictions], references=[references], lang="en")
    tfidf_score = get_tfidf_similarity([predictions], [references])
    gptscore_score = get_gpt_score([predictions], [references])
    
    data_dict = {
        "Rouge_1": rouge_score["rouge1"],
        "Rouge_2": rouge_score["rouge2"],
        "Rouge_L": rouge_score["rougeL"],
        "Rouge_Lsum": rouge_score["rougeLsum"],
        "Bleu": bleu_score['bleu'],
        "Meteor": meteor_score["meteor"],
        "BertScore_precision": bertscore_score["precision"][0],
        "BertScore_recall": bertscore_score["recall"][0],
        "BertScore_f1": bertscore_score["f1"][0],
        "TFIDF": tfidf_score,
        "GPTScore": gptscore_score
    }
    return data_dict

def guidedog_aggregation_result(results, metric, args):
    result = np.mean(results)
    eval_logger.info(f"[{metric}]: {result}")
    
    if metric == "Rouge_1": return result
    elif metric == "Rouge_2": return result
    elif metric == "Rouge_L": return result
    elif metric == "Rouge_Lsum": return result
    elif metric == "Bleu": return result
    elif metric == "Meteor": return result
    elif metric == "BertScore_precision": return result
    elif metric == "BertScore_recall": return result
    elif metric == "BertScore_f1": return result
    elif metric == "TFIDF": return result
    elif metric == "GPTScore": return result

def guidedog_rouge1(results, args): return guidedog_aggregation_result(results, "Rouge_1", args)
def guidedog_rouge2(results, args): return guidedog_aggregation_result(results, "Rouge_2", args)
def guidedog_rougeL(results, args): return guidedog_aggregation_result(results, "Rouge_L", args)
def guidedog_rougeLsum(results, args): return guidedog_aggregation_result(results, "Rouge_Lsum", args)
def guidedog_bleu(results, args): return guidedog_aggregation_result(results, "Bleu", args)
def guidedog_meteor(results, args): return guidedog_aggregation_result(results, "Meteor", args)
def guidedog_bertscore_precision(results, args): return guidedog_aggregation_result(results, "BertScore_precision", args)
def guidedog_bertscore_recall(results, args): return guidedog_aggregation_result(results, "BertScore_recall", args)
def guidedog_bertscore_f1(results, args): return guidedog_aggregation_result(results, "BertScore_f1", args)
def guidedog_tfidf(results, args): return guidedog_aggregation_result(results, "TFIDF", args)
def guidedog_gptscore(results, args): return guidedog_aggregation_result(results, "GPTScore", args)
