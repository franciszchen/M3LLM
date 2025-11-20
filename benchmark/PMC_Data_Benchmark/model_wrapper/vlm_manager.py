def get_model(model_name: str, model_path: str, args):
    model_name = model_name.lower()
    if model_name == "qwen2.5-vl":
        from model_wrapper.Qwen2_5.qwen2_5_eval import Qwen2_5_VL
        return Qwen2_5_VL(model_path, args)
    if model_name == "llava":
        from model_wrapper.LLaVA.llava_eval import Llava
        return Llava(model_path, args)

    if model_name == 'llava-med':
        from model_wrapper.LLava_Med.llava_med_eval import LLavaMed
        return LLavaMed(model_path, args)

    if model_name == 'huatuo':
        from model_wrapper.HuatuoGPT.huatuo import HuatuoGPT
        return HuatuoGPT(model_path, args)

    if model_name == 'internvl':
        from model_wrapper.InternVL.internvl_eval import InternVL
        return InternVL(model_path, args)

    if model_name == 'healthgpt':
        from model_wrapper.HealthGPT.HealthGPT_phi import HealthGPT
        return HealthGPT(model_path, args)
    if model_name == "medgemma":
        from model_wrapper.medgemma.medgemma_eval import MedGemma
        return MedGemma(model_path, args)

    else:
        raise ValueError(f"Model '{model_name}' not supported.")

