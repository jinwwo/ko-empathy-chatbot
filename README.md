## ko-empathy-chatbot
A chatbot model fine-tuned using QLoRA on sLLM

## Datasets
Source: [AI Hub 공감형 대화](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71305)

## Data preprocessing
- Check [Here](https://github.com/jinwwo/ko-chatbot/blob/0907ce6aca40888f0d1e1864d0d491d053da9ab2/data/preprocess.ipynb)
1. Extract utterances only
2. Convert to DatasetDict

## Training
- base model: [yanolja/EEVE-Korean-10.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-10.8B-v1.0)
- Epoch: 2
- LoRA r: 16
- LoRA alpha: 32
- learning rate: 2e-4

## Reference
- Thanks to [jwj7140](https://github.com/jwj7140/ko-medical-chat)
