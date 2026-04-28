import json

BASE_PROMPT = "Which object is present in the image scene?"


def guidedog_object_doc_to_visual(doc):
    return [doc["image"]]


def guidedog_object_doc_to_text(doc):
    question = BASE_PROMPT
    choices = "\n".join(doc["choices"])
    text = f"{question}\n{choices}"
    return f"{text}\nAnswer with the option's letter from the given choices directly."


def guidedog_object_process_result(doc, result):
    pred = result[0].split(".")[0].strip()
    if len(pred) > 1:
        pred = pred[0]
    answer = doc["answer"]
    image_id = doc["id"].split(".")[0]
    # print(pred, answer)
    
    return {"accuracy": {"pred": pred, "answer": answer, "image_id": image_id}}


def guidedog_object_aggregation_result(results):
    total_count = 0
    total_correct = 0
    for result in results:
        if result["pred"].lower().strip() == result["answer"].lower().strip():
            total_correct += 1
        total_count += 1
    return total_correct / total_count