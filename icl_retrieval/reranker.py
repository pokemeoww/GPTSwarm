import torch
import torch.nn as nn
#from transformers import AutoTokenizer, AutoModelForSequenceClassification
from modelscope import AutoTokenizer, AutoModelForSequenceClassification
from icl_retrieval.utils.device_utils import get_device

from transformers import BitsAndBytesConfig


class RerankerModel(nn.Module):
    def __init__(
            self,
            bert_model_path: str = None,
    ) -> None:
        super(RerankerModel, self).__init__()

        self.device = get_device()
        print(f"[RerankerModel] Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_path)

    #     bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(bert_model_path,
        #quantization_config=bnb_config
        )
        self.model = self.model.to(self.device)

        self.act = torch.nn.Sigmoid()

    def forward(self, **inputs):
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''

        scores = self.act(self.model(**inputs, return_dict=True).logits.view(-1, ).float())
        return scores

    def predict(self, query, demos, embed_model=None):

        pairs = []
        for demo_ in demos:
            pairs.append(
                [demo_["sample"]["input"], query]
            )
        sentence_a = [pair[0] for pair in pairs]
        sentence_b = [pair[1] for pair in pairs]

        # 批量编码句子对
        inputs = self.tokenizer(
            sentence_a,
            sentence_b,
            max_length=512,  # 最大长度（按需调整）
            padding='longest',  # 填充到最大长度
            truncation=True,  # 截断超过长度的部分
            return_tensors='pt',  # 返回PyTorch张量
            return_token_type_ids=True  # 需要token_type_ids
        )
        inputs = {k_: v_.to(self.device) for k_, v_ in inputs.items()}

        # reranker scores
        self.model.eval()
        with torch.no_grad():
            rerank_scores = self(
                **inputs
            )
        rerank_scores = rerank_scores.cpu().numpy().tolist()

        demos_w_scores = [(w, s) for w, s in zip(demos, rerank_scores)]
        rerank_demos = sorted(
            demos_w_scores,
            key=lambda x: x[1],
            reverse=True
        )
        rerank_demos_final = [w[0] for w in rerank_demos]
        rerank_scores_final = [w[1] for w in rerank_demos]

        return rerank_demos_final, rerank_scores_final
