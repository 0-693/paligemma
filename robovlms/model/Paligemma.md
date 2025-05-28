# Paligemmaģ�����

���ĵ���ϸ˵����Paligemmaģ�͵ļܹ�������RoboVLMs��Ŀ�е�ʵ�֣�����ģ���������ʼ�����̡��ؼ����Ժͷ����Ⱥ������ݡ�

## 1. ģ�ͼܹ�����

Paligemma��Google������һ����ģ̬�Ӿ�����ģ��(VLM)�����ܹ�����ͼ����ı����룬�������ı��������RoboVLMs��Ŀ�У�Paligemma����װΪ`RoboPaligemma`�࣬�̳���`BaseRoboVLM`���ࡣ

### 1.1 �������

Paligemmaģ����������Ҫ������ɣ�

1. **SigLIP�Ӿ�������**��������ͱ���ͼ������
2. **Gemma����ģ��**���������ı�����������ı����
3. **��ģ̬ͶӰ��**�������Ӿ�������ģ�ͣ�ʵ�ֶ�ģ̬�ں�

��`config.json`�ļ��п��Կ���ģ�͵Ļ������ã�

```json
{
  "hidden_size": 2048,
  "projection_dim": 2048,
  "text_config": {
    "hidden_size": 2048,
    "intermediate_size": 16384,
    "model_type": "gemma",
    "num_attention_heads": 8,
    "num_hidden_layers": 18,
    "num_image_tokens": 256,
    "num_key_value_heads": 1,
    "vocab_size": 257216
  },
  "vision_config": {
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "model_type": "siglip_vision_model",
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "num_image_tokens": 256,
    "patch_size": 14,
    "projection_dim": 2048
  }
}
```

## 2. RoboPaligemma��ʵ��

### 2.1 �ඨ��

`RoboPaligemma`�ඨ����`robovlms/model/backbone/robopaligemma.py`�ļ��У�

```python
class RoboPaligemma(BaseRoboVLM):
    @property
    def image_processor(self):
        return self.processor

    @property
    def hidden_size(self):
        return self.model.config.text_config.hidden_size

    @property
    def word_embedding(self):
        return self.model.language_model.model.embed_tokens

    @property
    def text_tower(self):
        return self.model.language_model.model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def model(self):
        return self.backbone

    def model_encode_images(self, images):
        image_outputs = self.model.vision_tower(images)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.model.multi_modal_projector(selected_image_feature)
        image_features = image_features / (self.model.config.hidden_size**0.5)
        return image_features
```

### 2.2 ���Ժͷ���

`RoboPaligemma`�ඨ���˶�����Է��������ڷ���ģ�͵ĸ��������

- **image_processor**������ͼ������������Ԥ��������ͼ��
- **hidden_size**������ģ�͵����ز��С�������ı�����
- **word_embedding**����������ģ�͵Ĵ�Ƕ���
- **text_tower**����������ģ�Ͳ���
- **vision_tower**�������Ӿ�ģ�Ͳ���
- **model**�����صײ�backboneģ��

`model_encode_images`�����Ǻ��Ĺ���֮һ�����ڱ���ͼ��������

1. ʹ���Ӿ�����������ͼ��
2. ��ȡ���һ�������״̬
3. ͨ����ģ̬ͶӰ�����Ӿ�����ͶӰ�����ı�������ͬ�Ŀռ�
4. ���������й�һ������

## 3. ģ�ͳ�ʼ������

### 3.1 �����ʼ��

`RoboPaligemma`�̳���`BaseRoboVLM`�࣬��ʼ��������Ҫ��`BaseRoboVLM.__init__`��������ɣ�

```python
def __init__(self, configs, train_setup_configs, ...):
    super().__init__()
    # ��ʼ���������ò���
    self.window_size = window_size
    self.use_obs_queries = use_obs_queries
    # ... ����������ʼ�� ...
    
    self.configs = configs
    self.model_name = configs["model"]
    self.model_config = json.load(
        open(
            os.path.join(
                self.configs["vlm"]["pretrained_model_name_or_path"], "config.json"
            ),
            "r",
        )
    )
    
    # ��ʼ��backbone��tokenizer
    self.tokenizer, self.backbone = self._init_backbone()
    self.tokenizer = update_tokenizer(self.tokenizer, self.configs["tokenizer"])
    # ... ������ʼ������ ...
```

### 3.2 Backbone��ʼ��

��`BaseRoboVLM`���У�`_init_backbone`���������ʼ��ģ�͵�backbone��tokenizer��

```python
def _init_backbone(self):
    tokenizer, model = build_vlm(self.configs["vlm"], self.configs["tokenizer"])
    if "Processor" in self.configs["tokenizer"]["type"]:
        self.processor = tokenizer
        self.tokenizer = self.processor.tokenizer
    else:
        self.tokenizer = tokenizer
    return self.tokenizer, model
```

### 3.3 VLM��������

`build_vlm`����������`robovlms/model/vlm_builder.py`�ļ��У���һ��ͨ�õ��Ӿ�����ģ�͹�����������֧�ֶ���VLMģ�͵ļ��أ�����Paligemma��LLaVA�ȡ���������ϸ�˽����������

#### 3.3.1 ��������˵��

```python
def build_vlm(vlm_config, tokenizer_config, precision="bf16"):
```

- **vlm_config**: �Ӿ�����ģ�͵������ֵ䣬�������¹ؼ��ֶΣ�
  - `pretrained_model_name_or_path`: Ԥѵ��ģ�͵�·����Hugging Faceģ����
  - `name`: ģ�����ƣ���"paligemma"��"llava"��
  - `type`: ģ�����ͣ�Ĭ��Ϊ"AutoModel"

- **tokenizer_config**: �ִ����������ֵ䣬������
  - `type`: �ִ�������
  - `pretrained_model_name_or_path`: �ִ�����·��

- **precision**: ģ�;������ã�Ĭ��Ϊ"bf16"��bfloat16��

#### 3.3.2 Paligemmaģ�ͼ�������

����Paligemmaģ�ͣ�����ִ�����²��裺

```python
if model_name == "paligemma":
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    # 1. ����ģ��
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # ʹ��bfloat16����
        device_map="cpu",           # ��ʼ���ص�CPU
        revision="bfloat16",       # ʹ��bfloat16�汾��Ȩ��
    )
    
    # 2. ���ش�����
    tokenizer = AutoProcessor.from_pretrained(model_path)
```

#### 3.3.3 ����ʾ��



```python
vlm_config = {
    "name": "paligemma",
    "type": "PaliGemmaForConditionalGeneration",
    "pretrained_model_name_or_path": ".vlms/paligemma-3b-pt-224"
}

tokenizer_config = {
    "type": "paligemma",
    "pretrained_model_name_or_path": ".vlms/paligemma-3b-pt-224"
}

# ����build_vlm����
tokenizer, model = build_vlm(vlm_config, tokenizer_config)
```

#### 3.3.4 ��RoboVLMs�е�ʹ��

��RoboVLMs��Ŀ�У�`build_vlm`����ͨ��ͨ��`BaseRoboVLM`���`_init_backbone`�������ã�

```python
def _init_backbone(self):
    # 1. ����ģ�ͺͷִ���
    tokenizer, model = build_vlm(self.configs["vlm"], self.configs["tokenizer"])
    
    # 2. ��������ķִ�������
    if "Processor" in self.configs["tokenizer"]["type"]:
        self.processor = tokenizer
        self.tokenizer = self.processor.tokenizer
    else:
        self.tokenizer = tokenizer
        
    return self.tokenizer, model
```

�������ʹ��ģ�͵ĳ�ʼ�����̸���ģ�黯�Ϳ���չ��֧�ֶ��ֲ�ͬ���͵��Ӿ�����ģ�͡�

## 4. ģ��������

### 4.1 SigLIP�Ӿ�������

SigLIP (Sigmoid Loss for Language Image Pre-training) ��Paligemma�е��Ӿ�������������Vision Transformer (ViT)�ܹ����������п��Կ�����

- ���ز��С��1152
- ע����ͷ����16
- ���ز�����27
- ͼ��token����256
- ͼ��patch��С��14

SigLIP��������ͼ��ָ��patches��Ȼ��ͨ�����Transformer������Ӿ�������

### 4.2 Gemma����ģ��

Gemma��Google�����Ŀ�Դ����ģ�ͣ���Paligemma����Ϊ�ı����������������������п��Կ�����

- ���ز��С��2048
- �м���С��16384
- ע����ͷ����8
- ���ز�����18
- �ʻ���С��257216

Gemma�������ı����벢�����ı������ͬʱҲ�������Ӿ������ںϺ�Ķ�ģ̬��ʾ��

### 4.3 ��ģ̬ͶӰ��

��ģ̬ͶӰ���������Ӿ�������ģ�͵Ĺؼ�����������Ӿ�����ͶӰ�����ı�������ͬ�Ŀռ䣬ʹ����ģ̬������������Ч�ںϡ���`model_encode_images`�����п��Կ�����ʹ�ã�

```python
image_features = self.model.multi_modal_projector(selected_image_feature)
image_features = image_features / (self.model.config.hidden_size**0.5)
```

ͶӰ�������������һ�������Ա����ı��������õ��ںϡ�

## 5. ģ��ʹ��ʾ��

��`robopaligemma.py`�ļ�������������չʾ����γ�ʼ����ʹ��`RoboPaligemma`ģ�ͣ�

```python
if __name__ == "__main__":
    configs = load_config(
        "configs/finetune_paligemma_cont-lstm-post_full-ft_text_vision_wd=0_hist=8_act=10_aug-shift_act-norm_lr-2e-5.json"
    )
    use_hand_rgb = False  # True
    model = RoboPaligemma(
        configs=configs,
        train_setup_configs=configs["train_setup"],
        fwd_head_configs=None,
        window_size=configs["window_size"],
        use_hand_rgb=use_hand_rgb,
        act_head_configs=configs["act_head"],
        fwd_pred_next_n=configs["fwd_pred_next_n"],
    )

    # ģ�Ͳ��Դ���
    bs, seq_len = 2, 2
    device = "cuda:0"
    vision_x = torch.zeros((bs, seq_len, 3, 224, 224), dtype=torch.float16).to(device)
    vision_gripper = torch.zeros((bs, seq_len, 3, 224, 224), dtype=torch.float16).to(
        device
    )
    lang_x = torch.ones((bs, 10), dtype=torch.long).to(device) * 100
    attention_mask = torch.ones((bs, 10)).bool().to(device)
    # ... �������Դ��� ...
```

## 6. ��RoboVLMs�е�Ӧ��

Paligemmaģ����RoboVLMs��Ŀ����Ҫ���ڻ������Ӿ����������磺

1. **�Ӿ����**���������˻����е��Ӿ�����
2. **ָ����ѭ**�������ı�ָ����Ӿ�����ִ����Ӧ����
3. **����Ԥ��**��Ԥ�������Ӧ��ִ�е���һ������

�����ļ�`finetune_paligemma_cont-lstm-post_full-ft_text_vision_wd=0_ws-8_act-10.json`�ж�����ģ�͵�ѵ����ʹ�ò�����

```json
{
  "robovlm_name": "RoboPaligemma",
  "model": "paligemma",
  "model_url": "https://huggingface.co/google/paligemma2-3b-pt-224",
  "vlm": {
    "type": "PaliGemmaForConditionalGeneration",
    "pretrained_model_name_or_path": ".vlms/paligemma-3b-pt-224",
    "name": "paligemma"
  },
  "tokenizer": {
    "type": "paligemma",
    "pretrained_model_name_or_path": ".vlms/paligemma-3b-pt-224"
  }
}
```

## 7. �ܽ�

Paligemmaģ����RoboVLMs��Ŀ�е�ʵ�ֿ����ܽ�Ϊ���¼����ؼ��㣺

1. ͨ��`RoboPaligemma`��̳�`BaseRoboVLM`���࣬ʵ���˶�Paligemmaģ�͵ķ�װ
2. ģ����SigLIP�Ӿ���������Gemma����ģ�ͺͶ�ģ̬ͶӰ����������������
3. ʹ��`build_vlm`������Hugging Face����Ԥѵ����Paligemmaģ�ͺʹ�����
4. ʵ����`model_encode_images`���������ڱ���ͼ�����������ı������ں�
5. �ṩ�˶�����Է��������ڷ���ģ�͵ĸ������
