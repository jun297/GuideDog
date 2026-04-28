import json

CLOSER_PROMPT = "A bounding box is an annotated rectangle surrounding an object. The edges of bounding boxes should touch the outermost pixels of the object that is being labeled. Given the two bounding boxes on the image, labeled by A and B, which bounding box is closer to the camera? Select from the following choices."
FARTHER_PROMPT = "A bounding box is an annotated rectangle surrounding an object. The edges of bounding boxes should touch the outermost pixels of the object that is being labeled. Given the two bounding boxes on the image, labeled by A and B, which bounding box is farther from the camera? Select from the following choices."


def guidedog_depth_doc_to_visual(doc):
    return [doc["image"]]


def guidedog_depth_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    global distance
    if lmms_eval_specific_kwargs['distance'] == "closer":
        question = CLOSER_PROMPT
    else:
        question = FARTHER_PROMPT
    choices = "\n".join([f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(doc["choices"])])
    text = f"{question}\n{choices}"
    return f"{text}\nAnswer with the option's letter from the given choices directly."


def guidedog_depth_aggregation_result(results):
    total_count = 0
    total_correct = 0
    for result in results:
        if result["pred"].lower().strip() == result["answer"].lower().strip():
            total_correct += 1
        total_count += 1
    return total_correct / total_count

############################################################

def guidedog_depth_closer_doc_to_target(doc):
    answer_index = doc['choices'].index(doc['closer'])
    return chr(ord('A') + answer_index)

def guidedog_depth_farther_doc_to_target(doc):
    answer_index = doc['choices'].index(doc['farther'])
    return chr(ord('A') + answer_index)


def guidedog_depth_closer_process_result(doc, result):
    pred = result[0].split(".")[0].strip()
    if len(pred) > 1:
        pred = pred[0]
    answer = f"{chr(ord('A') + doc['choices'].index(doc['closer']))}"
    image_id = doc["id"].split(".")[0]
    return {"accuracy": {"pred": pred, "answer": answer, "image_id": image_id}}

def guidedog_depth_farther_process_result(doc, result):
    pred = result[0].split(".")[0].strip()
    if len(pred) > 1:
        pred = pred[0]
    answer = f"{chr(ord('A') + doc['choices'].index(doc['farther']))}"
    image_id = doc["id"].split(".")[0]
    return {"accuracy": {"pred": pred, "answer": answer, "image_id": image_id}}
