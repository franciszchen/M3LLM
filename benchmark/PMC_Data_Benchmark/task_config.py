TASKS = {
    "puretext": {
        "json_path": "benchmark_data/puretextQA_benchmark.json",
        "image_root": "benchmark_images",
        "system_prompt": "You are a medical expert who is good at pure-text QA task. Please answer the following question carefully. Please do not answer with a single word. Do not overstate. Make the answer concise and precise. Answer with complete sentences."
    },
    "multi-choice": {
        "json_path": "benchmark_data/multiplechoiceVQA_benchmark.json",
        "image_root": "benchmark_images",
        "system_prompt": "You are a medical expert who is good at solving medical multiple-choice task. Please answer with option letter only."
    },
    "single-subimageVQA": {
        "json_path": "benchmark_data/single_subimageVQA_benchmark.json",
        "image_root": "benchmark_images",
        "system_prompt": "You are a medical expert who is good at single subimage VQA task. Please answer the question regarding a single sub-image. Please do not answer with a single word."
    },
    "bboxVQA": {
        "json_path": "benchmark_data/spatial_relation_benchmark.json",
        "image_root": "benchmark_images",
        "system_prompt": "You are a medical expert who is good at analyzing subimage spatial relationship. Use the compound image and sub-images to answer the question regarding the spatial relation of two sub-images."
    },
    "compoundVQA": {
        "json_path": "benchmark_data/compound-imageVQA_benchmark.json",
        "image_root": "benchmark_images",
        "system_prompt": "You are a medical expert who is good at analyzing compound medical image. Please answer the question regarding the given compound image."
    },
    "multisubimageVQA": {
        "json_path": "benchmark_data/multiplesubimageVQA_benchmark.json",
        "image_root": "benchmark_images",
        "system_prompt": "You are a medical expert who is good at multiple subimage VQA task. Please answer the question regarding the given multiple sub-images. Please do not answer with a single word."
    },
}
