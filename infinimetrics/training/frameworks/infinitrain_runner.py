import logging
from core.training_runner import TrainingRunner

class InfinitrainRunner(TrainingRunner):
    """Infinitrain training runner implementation (placeholder)"""
    def __init__(self, config_manager, gpu_monitor):
        super().__init__(config_manager, gpu_monitor)
        self.logger = logging.getLogger(self.__class__.__name__)

    def build_training_command(self):
        """Build Infinitrain training command"""
        self.logger.info("InfiniTrain runner is not implemented yet")
        raise NotImplementedError("InfiniTrain runner is not implemented yet")
    
    def parse_training_output(self, line, metrics):
        """Parse Infinitrain training output"""
        self.logger.debug(f"Parsing Infinitrain output: {line}")
        raise NotImplementedError("InfiniTrain runner is not implemented yet")
