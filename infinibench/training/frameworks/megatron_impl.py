import os
import re
import random
import subprocess
import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MegatronImpl:
    """
    Megatron-LM training executor.
    """

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
        "vlm": "pretrain_vlm.py",
    }

    DEFAULT_MODEL_CONFIG = {
        "num_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "seq_len": 1024,
        "vocab_size": 32000,
    }

    def __init__(
        self,
        config: Dict[str, Any],
        resolved_device_count: int,
        run_id: Optional[str] = None,
    ):
        self.config = config
        self.resolved_device_count = resolved_device_count
        self.logger = logger

        # Extract run_id and output paths
        self.run_id = run_id if run_id is not None else config.get("run_id", "unknown")
        self.logger.info(f"MegatronRunner using run_id: {self.run_id}")

        self.output_dir = Path(config.get("output_dir", "./output")) / "training"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training args
        self.train_args = config.get("train_args", {})
        self.base_model = config.get("model", "gpt")
        self.megatron_path = config.get("megatron_path", "")

        # Training mode
        self.training_mode = self._detect_training_mode(config)
        self.logger.info(
            f"Training mode: {self.training_mode}, base model: {self.base_model}"
        )

        # Parallel config
        parallel = self.train_args.get("parallel", {})
        self.tp = parallel.get("tp", 1)
        self.pp = parallel.get("pp", 1)
        self.dp = parallel.get("dp", 1)
        self.sp = parallel.get("sp", 0)

        # Basic training parameters
        self.mbs = self.train_args.get("mbs", 1)
        self.seq_len = self.train_args.get("seq_len", 1024)
        self.warmup_iterations = self.train_args.get("warmup_iterations", 0)

        # Calculate nproc_per_node
        self.nproc_per_node = self._calculate_nproc_per_node()
        self.master_port = random.randint(20000, 60000)

        # Regex patterns
        self.iter_pattern = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
        self.loss_pattern = re.compile(
            r"lm loss:\s*([+\-]?\d+(?:\.\d+)?(?:[Ee][+\-]?\d+)?)", re.IGNORECASE
        )
        self.elapsed_pattern = re.compile(
            r"elapsed time per iteration \(ms\):\s*([0-9]*\.?[0-9]+)", re.IGNORECASE
        )

        # Output files
        self.log_file = self.output_dir / f"{self.run_id}_train.log"
        self.loss_csv = self.output_dir / f"{self.run_id}_train_loss.csv"
        self.ppl_csv = self.output_dir / f"{self.run_id}_train_ppl.csv"
        self.throughput_csv = self.output_dir / f"{self.run_id}_train_throughput.csv"

        # Save the generated command
        self._last_command = None

    def _detect_training_mode(self, config: Dict[str, Any]) -> str:
        """Detect training mode from testcase or config."""
        testcase = self.config.get("_testcase", "").lower()
        if "lora" in testcase or self.config.get("use_lora"):
            return "lora"
        if "sft" in testcase or self.config.get("use_sft"):
            return "sft"
        if "pretrain" in testcase:
            return "pretrain"

        return "pretrain"

    def run(self) -> Dict[str, Any]:
        """Execute Megatron training."""
        cmd = self._build_command()
        self._last_command = " ".join(cmd)
        self.logger.info(f"Launching: {self._last_command}")

        # Metrics collection
        metrics = {
            "losses_by_iter": {},
            "throughput_by_iter": {},
            "last_seen_iter": None,
        }

        # Start process
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self._get_env(),
        )

        # Process output
        with open(self.log_file, "w") as f:
            for line in proc.stdout:
                line = line.rstrip("\n")
                self.logger.info(line)
                f.write(line + "\n")
                self._parse_output(line, metrics)

        proc.wait()

        if proc.returncode != 0:
            self._create_empty_csv_files()
            raise RuntimeError(
                f"Training failed (code {proc.returncode}). Log: {self.log_file}"
            )

        self._save_results(metrics)

        return {
            "log_file": str(self.log_file),
            "loss_csv": str(self.loss_csv),
            "ppl_csv": str(self.ppl_csv),
            "throughput_csv": str(self.throughput_csv),
            "metrics": metrics,
        }

    def _create_empty_csv_files(self):
        """Creates an empty CSV file"""
        with open(self.loss_csv, "w") as f:
            f.write("iteration,loss\n")
        with open(self.ppl_csv, "w") as f:
            f.write("iteration,ppl\n")
        with open(self.throughput_csv, "w") as f:
            f.write("iteration,throughput\n")
        self.logger.warning(f"Created empty CSV files due to training failure")

    def get_command(self) -> str:
        """Return the executed command."""
        return self._last_command or ""

    def _build_command(self) -> list:
        """Build torchrun command with all parameters."""
        train_script = self.MODEL_SCRIPT_MAP.get(self.base_model, "pretrain_gpt.py")
        if self.megatron_path:
            train_script = f"{self.megatron_path}/{train_script}"

        cmd = [
            "torchrun",
            f"--nproc_per_node={self.nproc_per_node}",
            f"--master_port={self.master_port}",
            train_script,
        ]

        # Add all argument groups
        cmd += self._add_parallel_args()
        cmd += self._add_model_args()
        cmd += self._add_training_args()
        cmd += self._add_precision_args()
        cmd += self._add_optimizer_args()
        cmd += self._add_lr_scheduler_args()
        cmd += self._add_dataset_args()
        cmd += self._add_training_mode_args()
        cmd += self._add_common_args()
        cmd += self._add_eval_checkpoint_args()

        extra_args = self.train_args.get("extra_args", [])
        if extra_args:
            cmd.extend(extra_args)

        return cmd

    def _add_parallel_args(self) -> list:
        """Add parallel configuration arguments."""
        args = [
            f"--tensor-model-parallel-size={self.tp}",
            f"--pipeline-model-parallel-size={self.pp}",
        ]
        if self.sp == 1:
            args.append("--sequence-parallel")
        return args

    def _add_model_args(self) -> list:
        """Add model configuration arguments from train_args."""
        args = [f"--seq-length={self.seq_len}"]

        for param, default in self.DEFAULT_MODEL_CONFIG.items():
            if param == "seq_len":
                continue
            value = self.train_args.get(param, default)
            args.append(f"--{param.replace('_', '-')}={value}")

        max_pos = self.train_args.get("max_position_embeddings", self.seq_len)
        args.append(f"--max-position-embeddings={max_pos}")

        return args

    def _add_precision_args(self) -> list:
        """Add precision arguments."""
        precision = self.train_args.get("precision", "fp16")
        if precision == "bf16":
            return ["--bf16"]
        elif precision == "fp16":
            return ["--fp16"]
        return []

    def _add_optimizer_args(self) -> list:
        """Add optimizer arguments."""
        args = [f"--optimizer={self.train_args.get('optimizer', 'adam')}"]

        weight_decay = self.train_args.get("weight_decay", 0.0)
        if weight_decay > 0:
            args.append(f"--weight-decay={weight_decay}")

        clip_grad = self.train_args.get("clip_grad", 1.0)
        if clip_grad > 0:
            args.append(f"--clip-grad={clip_grad}")

        optimizer = self.train_args.get("optimizer", "adam").lower()
        if optimizer in ["adam", "adamw"]:
            args.append(f"--adam-beta1={self.train_args.get('beta1', 0.9)}")
            args.append(f"--adam-beta2={self.train_args.get('beta2', 0.999)}")

        return args

    def _add_lr_scheduler_args(self) -> list:
        """Add learning rate scheduler arguments."""
        args = [f"--lr-decay-style={self.train_args.get('lr_scheduler', 'cosine')}"]

        min_lr = self.train_args.get("min_lr", 0.0)
        if min_lr > 0:
            args.append(f"--min-lr={min_lr}")

        if self.warmup_iterations > 0:
            args.append(f"--lr-warmup-iters={self.warmup_iterations}")

        return args

    def _add_dataset_args(self) -> list:
        """Add dataset arguments."""
        train_dataset = self.config.get("train_dataset", "mock")
        if train_dataset == "mock":
            return ["--mock-data", "--tokenizer-type", "NullTokenizer"]

        args = [f"--data-path={train_dataset}"]
        if val := self.config.get("validation_dataset"):
            args.append(f"--validation-data-path={val}")
        if test := self.config.get("test_dataset"):
            args.append(f"--test-data-path={test}")

        return args

    def _add_training_args(self) -> list:
        """Add basic training hyperparameters from train_args."""
        args = []

        # Micro-batch size
        mbs = self.train_args.get("mbs", 1)
        args.append(f"--micro-batch-size={mbs}")

        # Global batch size
        gbs = self.train_args.get("gbs")
        if gbs:
            args.append(f"--global-batch-size={gbs}")

        # Number of training iterations
        train_iters = self.train_args.get("train_iters")
        if train_iters:
            args.append(f"--train-iters={train_iters}")

        # Learning rate
        lr = self.train_args.get("lr")
        if lr:
            args.append(f"--lr={lr}")

        return args

    def _add_training_mode_args(self) -> list:
        """
        Add training mode specific arguments (LoRA/SFT).
        Dynamically detects if the current Megatron version supports these features.
        """
        args = []

        if self.training_mode == "pretrain":
            return args

        if self.training_mode == "sft":
            self.logger.info("SFT mode: adding --finetune flag")
            args.append("--finetune")

            pretrained_path = self.train_args.get("pretrained_path")
            if pretrained_path:
                args.append(f"--load={pretrained_path}")
                self.logger.info(f"Loading pretrained weights from {pretrained_path}")

            sft_lr = self.train_args.get("sft_lr")
            if sft_lr:
                args.append(f"--lr={sft_lr}")

            return args

        if self.training_mode == "lora":
            if self._check_lora_support():
                self.logger.info("LoRA mode: adding LoRA parameters")

                # LoRA specific parameters
                lora_rank = self.train_args.get("lora_rank", 8)
                lora_alpha = self.train_args.get("lora_alpha", 16)
                lora_dropout = self.train_args.get("lora_dropout", 0.1)
                lora_target_modules = self.train_args.get(
                    "lora_target_modules", ["q_proj", "v_proj"]
                )

                args.extend(
                    [
                        "--lora",
                        f"--lora-rank={lora_rank}",
                        f"--lora-alpha={lora_alpha}",
                        f"--lora-dropout={lora_dropout}",
                        f"--lora-target-modules={','.join(lora_target_modules)}",
                    ]
                )

                pretrained_path = self.train_args.get("pretrained_path")
                if pretrained_path:
                    args.append(f"--load={pretrained_path}")
            else:
                fallback = self.config.get("lora_fallback_mode", "error")
                if fallback == "finetune":
                    self.logger.warning(
                        f"LoRA not supported, falling back to --finetune as requested. "
                        f"Set 'lora_fallback_mode': 'error' to fail instead."
                    )
                    args.append("--finetune")
                    pretrained_path = self.train_args.get("pretrained_path")
                    if pretrained_path:
                        args.append(f"--load={pretrained_path}")
                else:
                    raise RuntimeError(
                        f"LoRA training mode requested but current Megatron-LM does not support LoRA. "
                        f"Megatron path: {self.megatron_path}"
                    )

        return args

    def _check_lora_support(self) -> bool:
        """
        Check if the current Megatron-LM installation supports LoRA.
        This can be extended based on actual LoRA support detection logic.
        """
        if self.config.get("force_lora_support", False):
            return True

        try:
            if self.megatron_path:
                lora_files = [
                    os.path.join(self.megatron_path, "megatron", "core", "lora"),
                    os.path.join(self.megatron_path, "megatron", "lora"),
                    os.path.join(self.megatron_path, "lora"),
                ]
                for path in lora_files:
                    if os.path.exists(path):
                        self.logger.info(f"LoRA support detected at {path}")
                        return True
        except Exception as e:
            self.logger.debug(f"Error checking LoRA support: {e}")

        return False

    def _add_common_args(self) -> list:
        """Add common Megatron arguments."""
        return [
            "--transformer-impl",
            "local",
            "--no-gradient-accumulation-fusion",
            "--no-persist-layer-norm",
            "--log-interval",
            "1",
            "--log-throughput",
        ]

    def _add_eval_checkpoint_args(self) -> list:
        """Add evaluation and checkpoint arguments."""
        args = []
        if eval_interval := self.train_args.get("eval_interval", 100):
            args.append(f"--eval-interval={eval_interval}")
            args.append(f"--eval-iters={self.train_args.get('eval_iters', 10)}")
        if save_interval := self.train_args.get("save_interval", 1000):
            args.append(f"--save-interval={save_interval}")
        return args

    def _get_env(self) -> Dict[str, str]:
        """Get environment variables for subprocess."""
        env = os.environ.copy()
        if device_ids := self.config.get("device", {}).get("device_ids"):
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))
            self.logger.info(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
        return env

    def _parse_output(self, line: str, metrics: Dict[str, Any]) -> None:
        """Parse Megatron output line."""
        # Update iteration
        m_iter = self.iter_pattern.search(line)
        if m_iter:
            try:
                metrics["last_seen_iter"] = int(m_iter.group(1))
            except Exception:
                pass

        # Extract loss
        m_loss = self.loss_pattern.search(line)
        if m_loss and metrics["last_seen_iter"] is not None:
            try:
                val = float(m_loss.group(1))
                if metrics["last_seen_iter"] > self.warmup_iterations:
                    metrics["losses_by_iter"][metrics["last_seen_iter"]] = val
            except Exception:
                pass

        # Extract throughput
        m_elapsed = self.elapsed_pattern.search(line)
        if m_elapsed and metrics["last_seen_iter"] is not None:
            try:
                elapsed_ms = float(m_elapsed.group(1))
                # tokens per iteration = mbs * seq_len * dp
                tokens_per_iter = self.mbs * self.seq_len * self.dp
                if elapsed_ms > 0:
                    throughput = (
                        tokens_per_iter / (elapsed_ms / 1000.0) / self.nproc_per_node
                    )
                    if metrics["last_seen_iter"] > self.warmup_iterations:
                        metrics["throughput_by_iter"][
                            metrics["last_seen_iter"]
                        ] = throughput
            except Exception:
                pass

    def _save_results(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to CSV."""
        # Save loss
        with open(self.loss_csv, "w") as f:
            f.write("iteration,loss\n")
            for it, val in metrics["losses_by_iter"].items():
                f.write(f"{it},{val}\n")

        # Save PPL
        with open(self.ppl_csv, "w") as f:
            f.write("iteration,ppl\n")
            for it, loss in metrics["losses_by_iter"].items():
                try:
                    ppl = float(math.exp(loss))
                except Exception:
                    ppl = float("inf")
                f.write(f"{it},{ppl}\n")

        # Save throughput
        with open(self.throughput_csv, "w") as f:
            f.write("iteration,throughput\n")
            for it, val in metrics["throughput_by_iter"].items():
                f.write(f"{it},{val}\n")

        self.logger.info(f"Results saved to {self.output_dir}")

    def _calculate_nproc_per_node(self) -> int:
        """Calculate number of processes per node using resolved device count."""
        desired_world = self.dp * self.tp * max(1, self.pp)
        return (
            min(desired_world, self.resolved_device_count)
            if self.resolved_device_count > 0
            else desired_world
        )
