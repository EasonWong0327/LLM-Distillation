import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import jsonlines
import os
from tqdm import tqdm
import re
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("medical_diagnosis.log")
    ]
)
logger = logging.getLogger(__name__)


class MedicalDiagnosisSystem:
    def __init__(self):
        """
        初始化医疗诊断系统，使用本地模型
        """
        # 检查 CUDA 是否可用
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            print(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Current GPU Memory Usage:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
        else:
            print("CUDA is not available. Running on CPU")

        # 使用本地模型路径
        model_path = r"/mnt/workspace/Qwen2.5-7B-Instruct"

        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 设置device_map
        if self.cuda_available:
            device_map = "auto"
            print("Using automatic device mapping")
        else:
            device_map = "cpu"
            print("Using CPU for model")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.cuda_available else torch.float32
        ).eval()

        # 打印模型位置信息
        print("\nModel device mapping:")
        if hasattr(self.model, 'hf_device_map'):
            for key, device in self.model.hf_device_map.items():
                print(f"{key}: {device}")

        # 设置生成参数
        self.generation_config = {
            "max_new_tokens": 1024,
            "top_p": 0.8,
            "temperature": 0.7,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "num_return_sequences": 1
        }

        # 如果使用GPU，再次打印内存使用情况
        if self.cuda_available:
            print(f"\nGPU Memory Usage after model loading:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

        print("Model initialization completed!")

    def create_prompt(self, case_data: Dict) -> str:
        """
        Create optimized medical diagnosis prompt
        """
        prompt = f"""你是一位经验丰富的临床医生，请对以下病例进行专业分析和诊断。

                病例信息：
                {case_data['feature_content']}

                分析要求：
                1. 系统地分析患者的主诉、现病史、体格检查和辅助检查结果
                2. 识别关键症状和体征，判断它们之间的相互关系
                3. 基于临床表现和检查结果，推理最可能的诊断
                4. 给出诊断依据，要点式列出支持该诊断的关键证据

                请在分析时特别注意：
                - 患者的年龄和性别特征
                - 症状出现的时间和发展过程
                - 体温、生命体征和体格检查的异常发现
                - 实验室检查结果的临床意义

                请仅输出以下JSON格式的诊断结果：
                {{
                    "reason": "诊断理由（以数字编号的形式列出关键证据点，如：1. xxx\\n2. xxx）",
                    "diseases": "最终诊断结果（请使用标准疾病名称）"
                }}

                确保输出是有效的JSON格式，不包含任何其他内容。请保持诊断的客观性和准确性。"""

        return prompt

    def extract_json_from_response(self, response: str) -> Dict:
        """
        Extract and parse JSON from model response with improved error handling

        Args:
            response: Raw response string from the model

        Returns:
            Parsed JSON as dictionary

        Raises:
            ValueError: If JSON extraction or parsing fails
        """
        logger.debug(f"Raw response to parse: {response}")

        # If response is empty or None, return a default response
        if not response or response.isspace():
            logger.warning("Empty response received from model")
            return {
                "reason": "模型未返回有效响应",
                "diseases": "处理失败"
            }

        # Strip non-JSON text that might appear before or after the JSON content
        # Look for ### or similar markers that might separate JSON from other text
        for separator in ["###", "---", "response:", "Response:", "结果:", "诊断结果:"]:
            if separator in response:
                # Try to get content after the separator
                parts = response.split(separator, 1)
                if len(parts) > 1:
                    response = parts[1].strip()

        # Try different patterns to extract JSON
        json_patterns = [
            r"```json\s*(.*?)\s*```",  # JSON in code blocks with json tag
            r"```\s*(.*?)\s*```",  # JSON in any code blocks
            r"({[\s\S]*?})",  # Any JSON-like structure with curly braces
            r'(\{"reason":[\s\S]*?"diseases":[\s\S]*?})'  # Specific pattern for our expected JSON structure
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        return json.loads(match.strip())
                    except json.JSONDecodeError:
                        continue

        # If no patterns worked, try the entire response
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # If we get here, try more aggressive fixes for common JSON issues

            # First attempt: Basic cleanup of common issues
            fixed_response = response.strip()
            # Replace single quotes with double quotes
            fixed_response = re.sub(r"'([^']*)'", r'"\1"', fixed_response)
            # Replace Chinese quotes with standard quotes
            fixed_response = fixed_response.replace('"', '"').replace('"', '"')
            fixed_response = fixed_response.replace('：', ':')
            # Add quotes around unquoted keys
            fixed_response = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', fixed_response)

            try:
                return json.loads(fixed_response)
            except json.JSONDecodeError:
                # Second attempt: Try to manually extract the fields
                logger.warning("Failed to parse JSON with standard methods, attempting manual extraction")

                reason_match = re.search(r'"?reason"?\s*:?\s*"([^"]*)"', response, re.DOTALL)
                diseases_match = re.search(r'"?diseases"?\s*:?\s*"([^"]*)"', response, re.DOTALL)

                if reason_match and diseases_match:
                    return {
                        "reason": reason_match.group(1).strip(),
                        "diseases": diseases_match.group(1).strip()
                    }

                # Last resort: Create a default response with the text content
                logger.error(f"Failed to extract valid JSON from response")
                # Extract what looks like the disease diagnosis
                diseases_candidates = re.findall(r'诊断结果[：:]\s*(.*?)(?:\n|$)', response)
                diseases = diseases_candidates[0] if diseases_candidates else "无法解析诊断结果"

                # Extract what looks like reasons
                reason_content = []
                numbered_points = re.findall(r'\d+\.\s*(.*?)(?:\n|$)', response)
                if numbered_points:
                    reason_content = [f"{i + 1}. {point}" for i, point in enumerate(numbered_points)]

                return {
                    "reason": "\n".join(reason_content) if reason_content else "无法解析诊断理由",
                    "diseases": diseases
                }

    def process_single_case(self, case_data: Dict) -> Dict:
        """
        Process a single medical case

        Args:
            case_data: Dictionary containing case information

        Returns:
            Dictionary with diagnosis results
        """
        case_id = case_data.get("id", "unknown")

        try:
            prompt = self.create_prompt(case_data)

            # Log memory usage before processing
            if self.cuda_available:
                start_mem = torch.cuda.memory_allocated() / 1024 ** 2
                logger.info(f"\nGPU Memory before processing case {case_id}: {start_mem:.2f} MB")

            # Encode input and move to GPU if available
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.cuda_available:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate response with error handling
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **self.generation_config
                    )

                # Decode response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print('-------0-----------')
                print(response)
                print('-------1-----------')
                # Extract only the part after the prompt
                response = response[len(prompt):].strip()
                print(response)
                print('-------2-----------')

                # Extract JSON from response
                result = self.extract_json_from_response(response)
                print(response)
                print('-------3-----------')

                # Validate required fields
                if not all(key in result for key in ["reason", "diseases"]):
                    missing_keys = [key for key in ["reason", "diseases"] if key not in result]
                    raise ValueError(f"JSON response missing required keys: {missing_keys}")

                return {
                    "id": case_id,
                    "reason": result["reason"],
                    "diseases": result["diseases"]
                }

            except torch.cuda.OutOfMemoryError:
                logger.error(f"CUDA out of memory for case {case_id}")
                if self.cuda_available:
                    torch.cuda.empty_cache()
                return self._create_error_result(case_id, "CUDA out of memory error")

        except Exception as e:
            logger.error(f"Error processing case {case_id}: {type(e).__name__} - {str(e)}")
            return self._create_error_result(case_id, f"{type(e).__name__} - {str(e)}")

        finally:
            # Log memory usage after processing
            if self.cuda_available:
                end_mem = torch.cuda.memory_allocated() / 1024 ** 2
                logger.info(f"GPU Memory after processing case {case_id}: {end_mem:.2f} MB")
                if 'start_mem' in locals():
                    logger.info(f"Memory change: {end_mem - start_mem:.2f} MB")

    def _create_error_result(self, case_id: str, error_message: str) -> Dict:
        """
        Create a standardized error result

        Args:
            case_id: Case identifier
            error_message: Error description

        Returns:
            Error result dictionary
        """
        return {
            "id": case_id,
            "reason": f"处理错误: {error_message}",
            "diseases": "处理失败"
        }

    def process_batch(self, input_file: str, output_file: str, start_index: int = 0):
        """
        批量处理病例
        """
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"Removed existing output file: {output_file}")

        with jsonlines.open(input_file) as reader:
            cases = list(reader)

        print(f"Total cases to process: {len(cases)}")
        print(f"Starting from index: {start_index}")

        results = []
        for i, case in enumerate(tqdm(cases[start_index:], desc="Processing cases")):
            try:
                print(f"\nProcessing case {i + start_index + 1}/{len(cases)}")

                # 在每个批次开始时清理GPU缓存
                if self.cuda_available and i % 10 == 0:
                    torch.cuda.empty_cache()
                    print(
                        f"Cleared GPU cache. Current memory usage: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

                result = self.process_single_case(case)
                results.append(result)

                with jsonlines.open(output_file, mode='a') as writer:
                    writer.write(result)

                if result["diseases"] == "处理失败":
                    print(f"Warning: Case {case['id']} processing failed")

            except Exception as e:
                print(f"Unexpected error processing case {case['id']}: {str(e)}")
                continue

        return results


def main():
    input_file = "20250208181531_camp_data_step_1_without_answer.jsonl"
    output_file = "baseline_result.jsonl"

    # 显示CUDA信息
    print("\nCUDA Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device count: {torch.cuda.device_count()}")

    system = MedicalDiagnosisSystem()

    try:
        results = system.process_batch(input_file, output_file, start_index=0)
        print(f"Successfully processed {len(results)} cases")

        failed_cases = sum(1 for r in results if r["diseases"] == "处理失败")
        success_cases = len(results) - failed_cases
        print(f"Success cases: {success_cases}")
        print(f"Failed cases: {failed_cases}")
        print(f"Results saved to {output_file}")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        print("Progress has been saved to the output file.")

    finally:
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\nFinal GPU Memory Usage:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")


if __name__ == "__main__":
    main()