import json
import os
import random
import string
from datetime import datetime

class ConfigManager:
    """Configuration manager class, unified handling of training configuration parameters"""
    
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_config()
        self.extract_parameters()
        self.validate_and_complete_config()
    
    def load_config(self):
        """Load configuration file"""
        with open(self.config_path, "r") as f:
            cfg = json.load(f)
        
        self.conf = cfg.get("config", {})
        self.raw_config = cfg

    def validate_and_complete_config(self):
        """Validate required fields and complete missing optional fields"""
        required_fields = ["framework", "model"]
        for field in required_fields:
            if field not in self.conf:
                raise ValueError(f"Required field '{field}' missing in config")

        if "run_id" not in self.conf:
            # Automatic generation run_id: train.{framework}.{task_type}.{timestamp}.{random_str}
            framework = self.conf.get("framework", "unknown")

            # Infer the type of task
            task_type = self.infer_task_type()

            # Generate a timestamp (YYYYMMDDHHMM)
            timestamp = datetime.now().strftime("%Y%m%d%H%M")

            # Generates an 8-bit random string
            random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

            self.conf["run_id"] = f"train.{framework}.{task_type}.{timestamp}.{random_str}"
            print(f"Generated run_id: {self.conf['run_id']}")

        # Generate or validate testcase
        if "testcase" not in self.conf:
            # The frame displays the name mapping
            framework_display_map = {
                "megatron": "MegatronLM",
                "infinitrain": "InfiniTrain"
            }

            framework = self.conf.get("framework", "unknown")
            task_type = self.infer_task_type()

            framework_display = framework_display_map.get(framework, framework.capitalize())
            self.conf["testcase"] = f"train.{framework_display}.{task_type}"
            print(f"Generated testcase: {self.conf['testcase']}")

    def infer_task_type(self):
        """Infer task type from configuration"""
        # Get from config or infer from data
        if "task_type" in self.conf:
            return self.conf["task_type"]

        # Inferring from the training data
        train_args = self.conf.get("train_args", {})

        # Check for LoRA-related parameters
        if train_args.get("lora_rank"):
            return "LoRA"

        # Check the training data set
        train_dataset = self.conf.get("train_dataset", "")
        if isinstance(train_dataset, str):
            dataset_lower = train_dataset.lower()
            if dataset_lower == "mock":
                return "Pretrain"
            elif any(keyword in dataset_lower for keyword in ['sft', 'instruction', 'finetune', 'fine-tune']):
                return "SFT"
            elif any(keyword in dataset_lower for keyword in ['rlhf', 'dpo', 'ppo', 'reward']):
                return "RLHF"

        # By default Pretrain
        return "Pretrain"
    
    def extract_parameters(self):
        """Extract and standardize configuration parameters"""
        # Test ID
        self.run_id = self.conf.get("run_id", "")
        self.testcase = self.conf.get("testcase", "")

        # Framework and Model configuration
        self.framework = self.conf.get("framework", "megatron").lower()
        self.model_name = self.conf.get("model", "gpt").lower()
        self.task_type = self.infer_task_type().lower()
        
        # Device configuration
        device_config = self.conf.get("device", {})
        self.gpu_platform = device_config.get("gpu_platform", "nvidia")
        self.device_ids = device_config.get("device_ids", [0])
        self.cpu_only = device_config.get("cpu_only", False)

        # Training parameters
        train_args = self.conf.get("train_args", {})
        self.train_args = train_args
        
        # Parallel configuration
        parallel = train_args.get("parallel", {})
        self.dp = int(parallel.get("dp", 1))
        self.tp = int(parallel.get("tp", 1))
        
        pp_raw = parallel.get("pp", 1)
        if isinstance(pp_raw, dict):
            self.pp_size = int(pp_raw.get("size", 1))
            self.pp_type = str(pp_raw.get("type", "default")).lower()
        else:
            self.pp_size = int(pp_raw)
            self.pp_type = "default"
        
        self.sp = int(parallel.get("sp", 0))
        
        # Training hyperparameters
        self.mbs = int(train_args.get("mbs", 1))
        self.gbs = int(train_args.get("gbs", max(1, self.mbs)))
        self.seq_len = int(train_args.get("seq_len", train_args.get("seq", 128)))
        self.lr = train_args.get("lr", 0.00015)
        self.train_iters = int(train_args.get("step", train_args.get("train_iters", train_args.get("train-iters", 10))))
        self.num_layers = int(train_args.get("num_layers", train_args.get("num-layers", 2)))
        self.hidden_size = int(train_args.get("hidden_size", train_args.get("hidden-size", 512)))
        self.num_attention_heads = int(train_args.get("num_attention_heads", train_args.get("num-attention-heads", 8)))
        self.max_position_embeddings = int(train_args.get("max_position_embeddings", train_args.get("max-position-embeddings", self.seq_len)))
        self.vocab_size = int(train_args.get("vocab_size", train_args.get("vocab-size", 128256)))
        
        self.precision = train_args.get("precision", "bf16")
        self.optimizer = train_args.get("optimizer", "adamw")
        self.weight_decay = float(train_args.get("weight_decay", 0.0))
        self.clip_grad = float(train_args.get("clip_grad", 0.0))
        self.beta1 = float(train_args.get("beta1", 0.9))
        self.beta2 = float(train_args.get("beta2", 0.95))

        # Learning rate scheduler
        self.lr_scheduler = train_args.get("lr_scheduler", "cosine")
        self.min_lr = float(train_args.get("min_lr", 0.0))
        self.warmup_ratio = train_args.get("warmup_ratio")
        self.warmup_iters = train_args.get("warmup_iters")
        self.warmup_samples = train_args.get("warmup_samples")
        self.lr_decay_iters = train_args.get("lr_decay_iters")
        self.lr_decay_samples = train_args.get("lr_decay_samples")
        
        # Evaluation and saving
        self.eval_interval = int(train_args.get("eval_interval", 100))
        self.eval_iters = int(train_args.get("eval_iters", 10))
        self.save_interval = int(train_args.get("save_interval", 1000))

        # Dataset configuration
        self.train_dataset = self.conf.get("train_dataset", None)
        self.validation_dataset = self.conf.get("validation_dataset", None)
        self.test_dataset = self.conf.get("test_dataset", None)
        
        # Runtime configuration
        self.output_dir = self.conf.get("output_dir", "./train")
        self.timeout_ms = train_args.get("timeout_ms", 10000)
        
        # Extract warmup and measured iterations
        self.warmup_iterations = int(train_args.get("warmup_iterations", train_args.get("warmup", 0)))
        self.measured_iterations = int(train_args.get("measured_iterations", self.train_iters))
    
    def get_model_config(self):
        """Get model configuration dictionary"""
        return {
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "vocab_size": self.vocab_size
        }
    
    def get_parallel_config(self):
        """Get parallel configuration"""
        return {
            "dp": self.dp,
            "tp": self.tp,
            "pp": {
                "size": self.pp_size,
                "type": self.pp_type
            },
            "sp": self.sp
        }
    
    def get_training_args(self):
        """Get training arguments"""
        return self.train_args

    def get_warmup_settings(self):
        """Intelligently get warmup settings, prioritize non-null values"""
        if self.warmup_samples is not None:
            return {"type": "samples", "value": self.warmup_samples}
        elif self.warmup_iters is not None:
            return {"type": "iters", "value": self.warmup_iters}
        elif self.warmup_ratio is not None:
            return {"type": "ratio", "value": self.warmup_ratio}
        else:
            return {"type": "ratio", "value": 0.03}  # Default

    def get_lr_decay_settings(self):
        """Intelligently get learning rate decay settings"""
        if self.lr_decay_samples is not None:
            return {"type": "samples", "value": self.lr_decay_samples}
        elif self.lr_decay_iters is not None:
            return {"type": "iters", "value": self.lr_decay_iters}
        else:
            return None
