import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
import os

class QwenTester:
    def __init__(self, model_path="Qwen/Qwen2.5-1.5B-Instruct"):
        # 关键设置：确保离线模式
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        # 使用绝对路径加载模型[2](@ref)
        if not os.path.exists(model_path.replace('/', os.sep)):
            # 如果传入的是HuggingFace ID，尝试查找本地缓存
            cache_path = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_path.replace('/', '--')}")
            if os.path.exists(cache_path):
                model_path = cache_path
            else:
                print(f"警告: 模型路径 {model_path} 不存在，请确保模型已下载")
        
        print(f"加载模型从: {model_path}")
        
        # 加载tokenizer和模型[1](@ref)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=True  # 强制使用本地文件[2](@ref)
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        
        # 验证token IDs在有效范围内[5](@ref)
        self._validate_token_ids()
    
    def _validate_token_ids(self):
        """验证特殊token ID是否在词汇表范围内"""
        vocab_size = self.model.config.vocab_size
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        
        print(f"词汇表大小: {vocab_size}")
        print(f"pad_token_id: {pad_token_id}")
        print(f"eos_token_id: {eos_token_id}")
        
        # 检查token ID是否有效
        if pad_token_id >= vocab_size:
            print(f"错误: pad_token_id {pad_token_id} 超出词汇表范围!")
            # 使用安全的fallback
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        if eos_token_id >= vocab_size:
            print(f"错误: eos_token_id {eos_token_id} 超出词汇表范围!")
            # 使用unk_token或第一个有效token
            if self.tokenizer.unk_token_id and self.tokenizer.unk_token_id < vocab_size:
                self.tokenizer.eos_token_id = self.tokenizer.unk_token_id
            else:
                self.tokenizer.eos_token_id = 0
    
    def test_generation(self, text, max_tokens=100, temperature=0.8, num_comps=1):
        """测试生成功能"""
        try:
            # 构建输入[1](@ref)
            messages = [
                {"role": "user", "content": text}
            ]
            
            # 使用chat template处理输入[5](@ref)
            input_ids = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 编码输入
            inputs = self.tokenizer(
                [input_ids], 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.model.device)
            
            print(f"输入形状: {inputs['input_ids'].shape}")
            print(f"输入token范围: {inputs['input_ids'].min()} - {inputs['input_ids'].max()}")
            
            # 关键：在CPU上运行以获取详细错误信息
            with torch.no_grad():
                # 先尝试在CPU上运行以调试
                cpu_inputs = {k: v.cpu() for k, v in inputs.items()}
                cpu_model = self.model.cpu()
                
                outputs = cpu_model.generate(
                    **cpu_inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    num_return_sequences=num_comps,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1  # 避免重复[5](@ref)
                )
                
                # 解码结果
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return generated_text
                
        except Exception as e:
            print(f"生成过程中出错: {e}")
            return None
    
    def debug_inputs(self, text):
        """调试输入处理"""
        messages = [{"role": "user", "content": text}]
        
        # 检查chat template
        template_output = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print("Chat Template 输出:")
        print(template_output)
        print("-" * 50)
        
        # 检查tokenization
        inputs = self.tokenizer(text, return_tensors="pt")
        print(f"Token IDs: {inputs['input_ids'][0][:10]}...")  # 只显示前10个
        print(f"Token转换: {[self.tokenizer.decode(tok) for tok in inputs['input_ids'][0][:10]]}")

def main():
    # 初始化测试器
    tester = QwenTester("Qwen/Qwen2.5-1.5B-Instruct")  # 或使用本地路径
    
    # 测试输入
    test_text = "请写一个简短的问候语"
    
    # 先调试输入
    print("=== 输入调试 ===")
    tester.debug_inputs(test_text)
    print("\n")
    
    # 测试生成
    print("=== 生成测试 ===")
    result = tester.test_generation(test_text, max_tokens=50, temperature=0.7)
    
    if result:
        print("生成成功!")
        print(f"结果: {result}")
    else:
        print("生成失败!")

if __name__ == "__main__":
    main()