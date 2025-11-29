# main_fixed.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
from collections import defaultdict, deque
import jieba
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime


class UsagePatternAnalyzer:
    """使用模式分析器"""

    def __init__(self, window_size=1000):
        self.request_history = deque(maxlen=window_size)
        self.user_profiles = defaultdict(lambda: {
            'query_patterns': [],
            'session_lengths': [],
        })
        print(f"初始化模式分析器，窗口大小: {window_size}")

    def add_request(self, request_data):
        """添加请求到历史记录"""
        self.request_history.append(request_data)

        user_id = request_data.get('user_id', 'default')
        self.user_profiles[user_id]['query_patterns'].append(
            request_data.get('query', '')
        )
        self.user_profiles[user_id]['session_lengths'].append(
            request_data.get('session_length', 1)
        )

    def analyze_patterns(self):
        """分析使用模式"""
        print(f"分析模式，当前历史记录数: {len(self.request_history)}")
        if len(self.request_history) < 3:
            print("历史记录不足，返回空模式")
            return {}

        patterns = {
            'temporal': self._detect_temporal_patterns(),
            'semantic': self._detect_semantic_clusters(),
            'user_specific': self._detect_user_patterns(),
        }
        return patterns

    def _detect_temporal_patterns(self):
        """检测时间模式"""
        timestamps = [req.get('timestamp', time.time()) for req in self.request_history]
        hours = [datetime.fromtimestamp(ts).hour for ts in timestamps]

        hour_counts = np.bincount(hours, minlength=24)
        peak_hours = np.argsort(hour_counts)[-2:]

        return {
            'peak_hours': peak_hours.tolist(),
            'hour_distribution': hour_counts.tolist()
        }

    def _detect_semantic_clusters(self):
        """基于语义相似度的请求聚类"""
        texts = [req.get('query', '') for req in self.request_history if req.get('query')]
        print(f"语义聚类文本数量: {len(texts)}")
        if len(texts) < 2:
            return {}

        try:
            vectorizer = TfidfVectorizer(max_features=20)
            features = vectorizer.fit_transform(texts).toarray()

            n_clusters = min(2, len(texts))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features)

            cluster_info = {}
            for i in range(n_clusters):
                cluster_texts = [texts[j] for j in range(len(texts)) if clusters[j] == i]
                if cluster_texts:
                    cluster_info[f'cluster_{i}'] = {
                        'size': len(cluster_texts),
                        'sample_queries': cluster_texts,
                    }

            return cluster_info
        except Exception as e:
            print(f"语义聚类失败: {e}")
            return {}

    def _detect_user_patterns(self):
        """检测用户特定模式"""
        user_patterns = {}
        for user_id, profile in self.user_profiles.items():
            if len(profile['query_patterns']) > 0:
                user_patterns[user_id] = {
                    'avg_session_length': np.mean(profile['session_lengths']) if profile['session_lengths'] else 0,
                    'unique_topics': len(set(profile['query_patterns'])),
                    'query_frequency': len(profile['query_patterns'])
                }
        return user_patterns


class PredictiveCacheManager:
    """预测性缓存管理器"""

    def __init__(self, cache_size=100):
        self.cache = {}
        self.access_stats = defaultdict(int)
        self.cache_size = cache_size
        print(f"初始化缓存管理器，缓存大小: {cache_size}")

    def get_cached_response(self, query, similarity_threshold=0.7):
        """获取缓存的响应"""
        best_match = None
        best_score = 0

        for cached_query in self.cache.keys():
            similarity = self._calculate_similarity(query, cached_query)
            if similarity > best_score and similarity >= similarity_threshold:
                best_score = similarity
                best_match = cached_query

        if best_match:
            print(f"✅ 缓存命中! 相似度: {best_score:.2f}")
            self.access_stats[best_match] += 1
            return self.cache[best_match]

        return None

    def cache_response(self, query, response):
        """缓存响应"""
        if len(self.cache) >= self.cache_size:
            self._evict_least_used()

        self.cache[query] = response

    def _calculate_similarity(self, query1, query2):
        """计算查询相似度"""
        words1 = set(jieba.cut(query1))
        words2 = set(jieba.cut(query2))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _evict_least_used(self):
        """淘汰最少使用的缓存项"""
        if not self.access_stats:
            key_to_remove = next(iter(self.cache.keys()))
        else:
            key_to_remove = min(self.access_stats.items(), key=lambda x: x[1])[0]

        del self.cache[key_to_remove]
        if key_to_remove in self.access_stats:
            del self.access_stats[key_to_remove]


class OptimizedQwenInference:
    """优化的Qwen推理引擎"""

    def __init__(self, model_path, dataset_path):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("加载模型和分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        self.model.eval()

        print("初始化分析器和缓存...")
        self.pattern_analyzer = UsagePatternAnalyzer()
        self.cache_manager = PredictiveCacheManager()

        # 加载数据集
        self.dataset = self._load_dataset()
        print(f"数据集加载完成，共 {len(self.dataset)} 条数据")

    def _load_dataset(self):
        """加载数据集 - 专门处理BUSTM的JSONL格式"""
        if not os.path.exists(self.dataset_path):
            print(f"数据集路径不存在: {self.dataset_path}")
            return self._create_test_data()

        all_data = []

        if os.path.isdir(self.dataset_path):
            print(f"扫描目录: {self.dataset_path}")
            # 递归扫描所有子目录
            for root, dirs, files in os.walk(self.dataset_path):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        print(f"处理文件: {file_path}")
                        data = self._load_bustm_jsonl_file(file_path)
                        all_data.extend(data)
                        print(f"  从 {file} 加载了 {len(data)} 条数据")
        else:
            # 单个文件
            all_data = self._load_bustm_jsonl_file(self.dataset_path)

        if len(all_data) == 0:
            print("未找到有效数据，使用测试数据")
            return self._create_test_data()

        return all_data

    def _load_bustm_jsonl_file(self, file_path):
        """专门加载BUSTM数据集的JSONL格式"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                        # BUSTM数据集格式: {"id":, "sentence1":, "sentence2":, "label":}
                        if isinstance(item, dict) and 'sentence1' in item:
                            # 将两个句子都作为查询数据
                            data.append({
                                'sentence1': item.get('sentence1', ''),
                                'sentence2': item.get('sentence2', ''),
                                'label': item.get('label', ''),
                                'id': item.get('id', line_num)
                            })
                    except json.JSONDecodeError as e:
                        print(f"  第{line_num}行JSON解析失败: {e}")
                        continue

        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")

        return data

    def _create_test_data(self):
        """创建测试数据"""
        print("创建测试数据...")
        test_data = [
            {"query": "你好，请介绍一下你自己", "user_id": "user_1"},
            {"query": "什么是人工智能", "user_id": "user_2"},
            {"query": "帮我写一个Python程序", "user_id": "user_1"},
            {"query": "机器学习有哪些应用", "user_id": "user_3"},
            {"query": "如何学习编程", "user_id": "user_2"},
            {"query": "推荐一些学习资源", "user_id": "user_1"},
            {"query": "解释一下深度学习", "user_id": "user_3"},
            {"query": "Python和Java有什么区别", "user_id": "user_2"},
            {"query": "如何提高编程能力", "user_id": "user_1"},
            {"query": "人工智能的未来发展", "user_id": "user_3"}
        ]
        return test_data

    def train_pattern_analyzer(self, num_samples=100):
        """训练模式分析器 - 使用真实的BUSTM数据"""
        print("训练模式分析器...")
        print(f"数据集总大小: {len(self.dataset)}")

        samples = self.dataset[:min(num_samples, len(self.dataset))]
        print(f"使用 {len(samples)} 个样本进行训练")

        processed_count = 0
        for i, sample in enumerate(samples):
            # 从BUSTM数据中提取句子作为查询
            queries = []
            if 'sentence1' in sample:
                queries.append(sample['sentence1'])
            if 'sentence2' in sample:
                queries.append(sample['sentence2'])
            # 如果是测试数据格式
            if 'query' in sample:
                queries.append(sample['query'])

            if queries:
                for j, query in enumerate(queries):
                    if query and query.strip():  # 确保查询非空
                        request_data = {
                            'query': query.strip(),
                            'user_id': f'user_{(i % 5) + 1}',  # 模拟5个用户
                            'timestamp': time.time() - (len(samples) - i) * 60 + j * 10,
                            'session_length': len(queries)
                        }
                        self.pattern_analyzer.add_request(request_data)
                        processed_count += 1

        print(f"成功处理 {processed_count} 个查询")

        patterns = self.pattern_analyzer.analyze_patterns()
        print(f"发现模式类型: {list(patterns.keys())}")

        # 打印模式详情
        for pattern_type, pattern_data in patterns.items():
            if pattern_data:  # 只打印非空模式
                print(f"  {pattern_type}: {pattern_data}")

    def generate_response(self, query, use_cache=True, max_length=256):
        """生成响应（带缓存优化）"""
        start_time = time.time()

        # 检查缓存
        if use_cache:
            cached_response = self.cache_manager.get_cached_response(query)
            if cached_response:
                return {
                    'response': cached_response,
                    'source': 'cache',
                    'latency': time.time() - start_time,
                    'cache_hit': True
                }

        # 记录请求用于模式分析
        request_data = {
            'query': query,
            'user_id': 'current_user',
            'timestamp': time.time(),
            'session_length': 1
        }
        self.pattern_analyzer.add_request(request_data)

        # 模型推理
        try:
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            response = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):],
                                             skip_special_tokens=True)

            # 缓存结果
            self.cache_manager.cache_response(query, response)

            result = {
                'response': response,
                'source': 'model',
                'latency': time.time() - start_time,
                'cache_hit': False
            }

            return result

        except Exception as e:
            return {
                'response': f"错误: {str(e)}",
                'source': 'error',
                'latency': time.time() - start_time,
                'cache_hit': False
            }

    def get_system_stats(self):
        """获取系统统计信息"""
        return {
            'cache_size': len(self.cache_manager.cache),
            'request_history': len(self.pattern_analyzer.request_history),
            'user_profiles': len(self.pattern_analyzer.user_profiles)
        }


def main():
    """主函数"""
    MODEL_PATH = r"D:\QwenQwen2.5-1.5B-Instruct"

    # 尝试不同的数据集路径
    dataset_paths = [
        r"D:\数据集\BUSTM",
        r"D:\数据集\FusedChat\raw\FusedChat\data",
        r"D:\数据集\FusedChat"
    ]

    # 选择第一个存在的路径
    dataset_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            print(f"使用数据集路径: {path}")
            break

    if dataset_path is None:
        print("❌ 没有找到可用的数据集路径")
        return

    print("🚀 初始化预测性优化系统...")

    try:
        system = OptimizedQwenInference(MODEL_PATH, dataset_path)

        # 训练模式分析器
        system.train_pattern_analyzer(num_samples=100)

        # 显示初始统计
        stats = system.get_system_stats()
        print("\n📊 初始系统统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # 交互式测试
        print("\n" + "=" * 50)
        print("🎯 预测性优化系统就绪!")
        print("输入 'stats' 查看系统状态")
        print("输入 'test' 运行测试查询")
        print("输入 'quit' 退出")
        print("=" * 50)

        # 从实际数据中提取一些示例查询
        example_queries = []
        if len(system.dataset) > 0:
            for i in range(min(5, len(system.dataset))):
                sample = system.dataset[i]
                if 'sentence1' in sample:
                    example_queries.append(sample['sentence1'])
                if 'sentence2' in sample and len(example_queries) < 5:
                    example_queries.append(sample['sentence2'])

        # 如果没找到示例，使用默认测试查询
        if not example_queries:
            example_queries = [
                "你好",
                "介绍一下人工智能",
                "什么是机器学习",
                "帮我写代码",
                "谢谢"
            ]

        while True:
            try:
                user_input = input("\n👤 用户: ").strip()

                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'stats':
                    stats = system.get_system_stats()
                    print("\n📊 系统统计:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue
                elif user_input.lower() == 'test':
                    # 运行测试查询
                    print("运行测试查询...")
                    for query in example_queries:
                        print(f"\n测试查询: {query}")
                        result = system.generate_response(query)
                        print(f"响应 ({result['source']}): {result['response']}")
                        print(f"延迟: {result['latency']:.2f}s")
                    continue
                elif not user_input:
                    continue

                # 生成响应
                result = system.generate_response(user_input)

                print(f"\n🤖 助手 ({result['source']}): {result['response']}")
                print(f"⏱️  延迟: {result['latency']:.2f}s")

            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as e:
                print(f"错误: {e}")

    except Exception as e:
        print(f"系统初始化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    input("\n按Enter键退出...")