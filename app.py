import torch
from transformers import AutoTokenizer
from palm_rlhf_pytorch import PaLM
import gradio as gr

def generate(prompt, seq_len=128, temperature=0.8, filter_thres=0.9):
    #device = torch.device("cpu")

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda:0")  # Sceglie la prima GPU
    else:
        device = torch.device("cpu")  # Sceglie la CPU se CUDA non è disponibile o non ci sono GPU

    num_tokens = 50304
    dim = 2048
    depth = 16
    dim_head = 128
    heads = 8
    flash_attn = True

    # model = PaLM(
    #     num_tokens=num_tokens, dim=dim, depth=depth, dim_head=dim_head, heads=heads, flash_attn=flash_attn
    # ).to(device).eval()

    """ codice  per la versione 410M"""
    model = PaLM(
        num_tokens=50304, dim=1024, depth=24, dim_head=128, heads=8, flash_attn=False, qk_rmsnorm = False,
    ).to(device).eval()


    """   codice per la  versione  a 1B   ##BUG: non funziona  i valori sono sbagliati """
    # model = PaLM(
    #     num_tokens=50304,  # confermato dal tuo output
    #     dim=2048,  # confermato dal tuo output
    #     depth=24,  # questo valore è stato mantenuto dal tuo codice originale
    #     dim_head=128,  # questo valore è stato mantenuto dal tuo codice originale
    #     heads=16,  # calcolato come dim / dim_head
    #     flash_attn=False,  # questo valore è stato mantenuto dal tuo codice originale
    #     qk_rmsnorm=False,  # questo valore è stato mantenuto dal tuo codice originale
    # ).to(device).eval()

    checkpoint = torch.load('./palm_410m_8k_v0.pt', map_location=device)
    # checkpoint = torch.load('./palm_1B_8k_v0.pt', map_location=device)
    model.load_state_dict(checkpoint)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    encoded_text = tokenizer(prompt, return_tensors="pt")

    output_tensor = model.generate(
        seq_len=seq_len,
        prompt=encoded_text["input_ids"].to(device),
        temperature=temperature,
        filter_thres=filter_thres,
        pad_value=0.0,
        eos_token=tokenizer.eos_token_id,
        return_seq_without_prompt=False,
        use_tqdm=True,
    )

    decoded_output = tokenizer.batch_decode(output_tensor, skip_special_tokens=True)

    return decoded_output

iface = gr.Interface(
    fn=generate,
    title="PaLM",
    description="Open-source PaLM demo.", 
    inputs="text", 
    outputs="text",
    # seq_len=gr.Slider(minimum=1, maximum=8192, step=1, default=32, label="Sequence Length"),
    # temperature=gr.Slider(minimum=0.0, maximum=1.0, step=0.01, default=0.8, label="Temperature"),
    # filter_thres=gr.Slider(minimum=0.0, maximum=1.0, step=0.01, default=0.9, label="Filter Threshold"),
)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint = torch.load('./palm_1B_8k_v0.pt', map_location=device)
# for key in checkpoint.keys():
#     print(key)


iface.launch()


