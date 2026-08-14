import random
import re
import subprocess
import logging

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

from core.training_runner import TrainingRunner

class MegatronRunner(TrainingRunner):
    """Megatron training runner implementation"""
    
    MODEL_SCRIPT_MAP = {
        "gpt": "pretrain_gpt.py",
        "bert": "pretrain_bert.py",
        "llama": "pretrain_gpt.py",
        "t5": "pretrain_t5.py",
        "ict": "pretrain_ict.py",
        "mamba": "pretrain_mamba.py",
        "retro": "pretrain_retro.py",
        "vision_classify": "pretrain_vision_classify.py",
        "vision_dino": "pretrain_vision_dino.py",
        "vision_inpaint": "pretrain_vision_inpaint.py",
        "vlm": "pretrain_vlm.py"
    }

    def __init__(self, config_manager, gpu_monitor):
        super().__init__(config_manager, gpu_monitor)
        self.nproc_per_node = self._calculate_nproc_per_node()
        self.master_port = self._pick_random_port()
        
        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # Compile regex patterns
        self.iter_pattern = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
        self.loss_pattern = re.compile(r"lm loss:\s*([+\-]?\d+(?:\.\d+)?(?:[Ee][+\-]?\d+)?)", re.IGNORECASE)
        self.elapsed_pattern = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
    
    def _pick_random_port(self, low=20000, high=60000):
        return random.randint(low, high)
    
    def _calculate_nproc_per_node(self):
        """Calculate number of processes per node"""
        desired_world = max(1, self.config.dp * self.config.tp * max(1, self.config.pp_size))
        if _TORCH_AVAILABLE:
            available_gpus = torch.cuda.device_count()
            return min(desired_world, available_gpus)
        else:
            return desired_world
    
    def build_training_command(self):
        """Build Megatron training command"""
        train_script = self.MODEL_SCRIPT_MAP.get(self.config.model_name, self.MODEL_SCRIPT_MAP["gpt"])
        
        # Set CUDA_VISIBLE_DEVICES if device_ids specified
        env_vars = {}
        if hasattr(self.config, 'device_ids') and self.config.device_ids:
            env_vars['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.config.device_ids))

        # Build torchrun command
        torchrun_cmd = [
            "torchrun", 
            f"--nproc_per_node={self.nproc_per_node}", 
            f"--master_port={self.master_port}"
        ]
        
        # Build Megatron arguments
        megatron_args = [train_script]
        
        # Add parallel configuration
        megatron_args += [
            f"--tensor-model-parallel-size={self.config.tp}",
            f"--pipeline-model-parallel-size={self.config.pp_size}",
        ]
        
        # Add training hyperparameters
        megatron_args += [
            f"--micro-batch-size={self.config.mbs}",
            f"--global-batch-size={self.config.gbs}",
            f"--seq-length={self.config.seq_len}",
            f"--lr={self.config.lr}",
            f"--train-iters={self.config.train_iters}",
            f"--num-layers={self.config.num_layers}",
            f"--hidden-size={self.config.hidden_size}",
            f"--num-attention-heads={self.config.num_attention_heads}",
            f"--max-position-embeddings={self.config.max_position_embeddings}",
            f"--vocab-size={self.config.vocab_size}",
        ]
        
        # Add precision configuration
        precision_map = {
            "bf16": "--bf16",
            "fp16": "--fp16",
            "fp32": "",
            "mixed": "--mixed-precision"
        }

        if self.config.precision in precision_map:
            arg = precision_map[self.config.precision]
            if arg:
                megatron_args.append(arg)

        # Add optimizer parameters
        if self.config.optimizer.lower() in ["adam", "sgd"]:
            megatron_args.append(f"--optimizer={self.config.optimizer.lower()}")
        else:
            self.logger.warning(f"Optimizer '{self.config.optimizer}' not supported by Megatron, using 'adam' instead")
            megatron_args.append("--optimizer=adam")

        if self.config.weight_decay > 0:
            megatron_args.append(f"--weight-decay={self.config.weight_decay}")

        if self.config.clip_grad > 0:
            megatron_args.append(f"--clip-grad={self.config.clip_grad}")
        
        if self.config.optimizer.lower() == "adam" or self.config.optimizer.lower() == "adamw":
            megatron_args.append(f"--adam-beta1={self.config.beta1}")
            megatron_args.append(f"--adam-beta2={self.config.beta2}")

        # Add learning rate scheduler
        scheduler_map = {
            "cosine": "cosine",
            "linear": "linear",
            "constant": "constant"
        }

        if self.config.lr_scheduler in scheduler_map:
            megatron_args.append(f"--lr-decay-style={scheduler_map[self.config.lr_scheduler]}")

        if self.config.min_lr > 0:
            megatron_args.append(f"--min-lr={self.config.min_lr}")

        # Warmup configuration
        warmup_settings = self.config.get_warmup_settings()
        if warmup_settings["type"] == "ratio":
            megatron_args.append(f"--lr-warmup-fraction={warmup_settings['value']}")
        elif warmup_settings["type"] == "iters":
            megatron_args.append(f"--lr-warmup-iters={warmup_settings['value']}")
        elif warmup_settings["type"] == "samples":
            megatron_args.append(f"--lr-warmup-samples={warmup_settings['value']}")

        # Learning rate decay settings
        lr_decay_settings = self.config.get_lr_decay_settings()
        if lr_decay_settings:
            if lr_decay_settings["type"] == "iters":
                megatron_args.append(f"--lr-decay-iters={lr_decay_settings['value']}")
            elif lr_decay_settings["type"] == "samples":
                megatron_args.append(f"--lr-decay-samples={lr_decay_settings['value']}")

        # Add data configuration
        if self.config.train_dataset is None or (isinstance(self.config.train_dataset, str) and self.config.train_dataset.lower() == "mock"):
            tokenizer_type = "NullTokenizer" 
            megatron_args += ["--mock-data", "--tokenizer-type", "NullTokenizer", "--vocab-size", str(self.config.vocab_size)]
        else:
            megatron_args += [f"--data-path={self.config.train_dataset}"]
            if self.config.validation_dataset:
                megatron_args += [f"--validation-data-path={self.config.validation_dataset}"]
            if self.config.test_dataset:
                megatron_args += [f"--test-data-path={self.config.test_dataset}"]
            if not self.config.validation_dataset and not self.config.test_dataset:
                megatron_args += ["--split", "99,1,0"]
        
        # Add common parameters
        megatron_args += [
            "--transformer-impl", "local",
            "--no-gradient-accumulation-fusion",
            "--no-persist-layer-norm",
            "--log-interval", "1",
            "--log-throughput"
        ]

        # Add evaluation configuration
        if self.config.eval_interval > 0:
            megatron_args.append(f"--eval-interval={self.config.eval_interval}")
            megatron_args.append(f"--eval-iters={self.config.eval_iters}")

        # Add checkpoint saving
        if self.config.save_interval > 0:
            megatron_args.append(f"--save-interval={self.config.save_interval}")
        
        if self.config.sp == 1:
            megatron_args.append("--sequence-parallel")
        
        # Combine commands
        cmd = torchrun_cmd + megatron_args

        # Set environment variables if needed
        if env_vars:
            self.logger.info(f"Setting environment variables: {env_vars}")

        return cmd
    
    def parse_training_output(self, line, metrics):
        """Parse Megatron training output"""
        # Update iteration number
        m_iter = self.iter_pattern.search(line)
        if m_iter:
            try:
                metrics['last_seen_iter'] = int(m_iter.group(1))
            except Exception:
                metrics['last_seen_iter'] = None
        
        # Extract loss
        m_loss = self.loss_pattern.search(line)
        if m_loss and metrics['last_seen_iter'] is not None:
            try:
                val = float(m_loss.group(1))
                if metrics['last_seen_iter'] > self.config.warmup_iterations:
                    metrics['losses_by_iter'][metrics['last_seen_iter']] = val
            except Exception:
                pass
        
        # Extract throughput
        m_elapsed = self.elapsed_pattern.search(line)
        if m_elapsed and metrics['last_seen_iter'] is not None:
            try:
                elapsed_ms = float(m_elapsed.group(1))
                tokens_per_iter_total = self.config.mbs * self.config.seq_len * self.nproc_per_node
                if elapsed_ms > 0:
                    throughput_gpu_normalized = tokens_per_iter_total / (elapsed_ms / 1000.0) / self.nproc_per_node
                    if metrics['last_seen_iter'] > self.config.warmup_iterations:
                        metrics['throughput_by_iter'][metrics['last_seen_iter']] = throughput_gpu_normalized
            except Exception:
                pass
