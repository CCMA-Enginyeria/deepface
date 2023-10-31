from deepface import DeepFace


def represent(img_path, model_name, detector_backend, enforce_detection, align):
    result = {}
    embedding_objs = DeepFace.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )
    result["results"] = embedding_objs
    return result


def verify(
    img1_path, img2_path, model_name, detector_backend, distance_metric, enforce_detection, align
):
    obj = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        align=align,
        enforce_detection=enforce_detection,
    )
    return obj


def analyze(img_path, actions, detector_backend, enforce_detection, align):
    result = {}
    demographies = DeepFace.analyze(
        img_path=img_path,
        actions=actions,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )
    result["results"] = demographies
    return result

def find(
    img_path,
    db_path,
    model_name,
    distance_metric,
    enforce_detection,
    detector_backend,
    align,
    normalization,
    silent,
):
    obj = DeepFace.find(
        img_path=img_path,
        db_path=db_path,
        model_name=model_name,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        normalization=normalization,
        silent=silent
    )
    def df_to_json(df):
        return df.iloc[0].to_json()
  
    finds = []
    for df in obj:
        if len(df) > 0:
            finds.append(df.iloc[0].to_dict())
    # We double all numbers using map()
    # finds = list(map(df_to_json, obj))
    result = {}
    result["results"] = finds
    return result
